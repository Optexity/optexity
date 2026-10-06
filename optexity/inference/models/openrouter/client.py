"""OpenRouter chat-completions client with per-model retries and fallback.

For each model in the chain:
- temporary errors (408, 429, 5xx, timeouts, connection errors) are retried,
  then the next model is tried;
- output that doesn't match the schema is retried, then the next model;
- model-specific rejections (context too long, moderation) go to the next model;
- auth, out-of-credits and bad-request errors are raised immediately.
"""

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import httpx
import openai
from pydantic import BaseModel

from .core import (
    Attempt,
    ChainEntry,
    Citation,
    LLMAllModelsFailed,
    LLMAuthError,
    LLMBadRequest,
    LLMConfig,
    LLMConfigError,
    LLMError,
    LLMInvalidOutput,
    LLMModelRejected,
    LLMOutOfCredits,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Tier,
    ToolCall,
    Usage,
    WebSearch,
    resolve_chain,
)
from .messages import inspect_messages, parse_structured, to_strict_schema

logger = logging.getLogger(__name__)

_DEFAULT = object()  # reasoning not given: use the tier's default
_FATAL = (LLMAuthError, LLMOutOfCredits, LLMBadRequest, LLMConfigError)
_TRANSIENT = (LLMRateLimited, LLMUnavailable, LLMTimeout)
# 400s that another model could still serve: the upstream provider rejected the
# request ("Provider returned error", e.g. a schema it can't compile) or the
# context is too long. OpenRouter's own 400s (bad model ID, bad params) stay fatal.
_MODEL_SPECIFIC = re.compile(
    r"^Provider returned error|context|too long|maximum.*tokens|token limit", re.I
)

# _run yields ("send", request kwargs, timeout) or ("sleep", seconds).
_Step = tuple


class OpenRouterClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ):
        if not config.api_key:
            raise LLMConfigError("OPENROUTER_API_KEY is not set")
        self.config = config
        self._http_client = http_client
        self._async_http_client = async_http_client
        self._sync: openai.OpenAI | None = None
        self._async: openai.AsyncOpenAI | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | Tier | None = None,
        schema: type[BaseModel] | None = None,
        schema_override: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        web_search: WebSearch | None = None,
        needs: set[str] | frozenset[str] = frozenset(),
        reasoning: Any = _DEFAULT,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        feature: str | None = None,
        session_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Send ``messages``, falling back through models as needed.

        ``model`` is an OpenRouter model ID, a ``Tier``, or ``None`` for the
        configured default; see ``resolve_chain``. ``schema`` requests structured
        output and fills ``LLMResult.parsed``; ``schema_override`` sends that JSON
        schema as-is (non-strict) instead of the one generated from ``schema``.
        ``tools`` (OpenAI function definitions) and ``tool_choice`` enable tool
        calling; calls come back in ``LLMResult.tool_calls``. ``stream`` streams
        the response and returns the same ``LLMResult``; the timeout then limits
        the gap between chunks instead of the whole response, so long outputs
        aren't cut off. ``web_search`` lets the model search the web; sources
        come back in ``LLMResult.citations``. ``timeout`` overrides the tier's
        per-attempt timeout.
        ``feature`` and ``session_id`` tag the request for usage reporting.
        """
        if self._sync is None:
            self._sync = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                max_retries=0,
                http_client=self._http_client,
            )
        run = self._run(
            messages, model, schema, schema_override, tools, tool_choice, stream, web_search, needs, reasoning, max_tokens,
            temperature, timeout, feature, session_id, extra_body,
        )  # fmt: skip
        step = next(run)
        while True:
            try:
                if step[0] == "sleep":
                    time.sleep(step[1])
                    step = run.send(None)
                    continue
                try:
                    resp = self._sync.chat.completions.create(
                        **step[1], timeout=step[2]
                    )
                    if step[1].get("stream"):
                        acc = _StreamAccumulator()
                        for chunk in resp:
                            acc.add(chunk)
                        resp = acc.result()
                except (openai.OpenAIError, httpx.HTTPError) as exc:
                    step = run.throw(exc)
                else:
                    step = run.send(resp)
            except StopIteration as done:
                return done.value

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | Tier | None = None,
        schema: type[BaseModel] | None = None,
        schema_override: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        web_search: WebSearch | None = None,
        needs: set[str] | frozenset[str] = frozenset(),
        reasoning: Any = _DEFAULT,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        feature: str | None = None,
        session_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Async version of ``complete``."""
        if self._async is None:
            self._async = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                max_retries=0,
                http_client=self._async_http_client,
            )
        run = self._run(
            messages, model, schema, schema_override, tools, tool_choice, stream, web_search, needs, reasoning, max_tokens,
            temperature, timeout, feature, session_id, extra_body,
        )  # fmt: skip
        step = next(run)
        while True:
            try:
                if step[0] == "sleep":
                    await asyncio.sleep(step[1])
                    step = run.send(None)
                    continue
                try:
                    resp = await self._async.chat.completions.create(
                        **step[1], timeout=step[2]
                    )
                    if step[1].get("stream"):
                        acc = _StreamAccumulator()
                        async for chunk in resp:
                            acc.add(chunk)
                        resp = acc.result()
                except (openai.OpenAIError, httpx.HTTPError) as exc:
                    step = run.throw(exc)
                else:
                    step = run.send(resp)
            except StopIteration as done:
                return done.value

    def _run(
        self,
        messages: list[dict[str, Any]],
        model: str | Tier | None,
        schema: type[BaseModel] | None,
        schema_override: dict | None,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        stream: bool,
        web_search: WebSearch | None,
        needs: set[str] | frozenset[str],
        reasoning: Any,
        max_tokens: int | None,
        temperature: float | None,
        timeout: float | None,
        feature: str | None,
        session_id: str | None,
        extra_body: dict[str, Any] | None,
    ) -> Generator[_Step, Any, LLMResult]:
        # Retry/fallback logic shared by complete() and acomplete(): it yields what
        # to do next, and the caller performs the request or sleep.
        msg_needs, est_tokens = inspect_messages(messages)
        all_needs = set(needs) | msg_needs
        response_format, aliases = None, {}
        if schema is not None:
            all_needs.add("json_schema")
            if schema_override is not None:
                json_schema, strict = schema_override, False
            else:
                json_schema, aliases = to_strict_schema(schema)
                strict = True
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^a-zA-Z0-9_-]", "_", schema.__name__)[:64],
                    "strict": strict,
                    "schema": json_schema,
                },
            }
        if tools:
            all_needs.add("tools")
        chain = resolve_chain(
            self.config,
            model,
            needs=frozenset(all_needs),
            est_tokens=est_tokens,
        )
        body = self._body(
            feature, session_id, schema is not None or bool(tools), all_needs
        )
        if web_search is not None:
            body.setdefault("plugins", []).append(web_search.plugin())
        body.update(extra_body or {})

        started = time.monotonic()
        attempts: list[Attempt] = []
        errors: list[LLMError] = []
        usage = Usage()
        for index, entry in enumerate(chain):
            request: dict[str, Any] = {
                "model": entry.model,
                "messages": messages,
                "extra_body": dict(body),
            }
            effort = entry.reasoning if reasoning is _DEFAULT else reasoning
            if effort:
                request["extra_body"]["reasoning"] = {"effort": effort, "exclude": True}
            if response_format is not None:
                request["response_format"] = response_format
            if stream:
                request["stream"] = True
            if tools:
                request["tools"] = tools
                if tool_choice is not None:
                    request["tool_choice"] = tool_choice
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            if temperature is not None:
                request["temperature"] = temperature
            attempt_timeout = timeout or entry.timeout

            transient_left = self.config.transient_retries
            invalid_left = self.config.invalid_output_retries
            while True:
                t0 = time.monotonic()
                try:
                    try:
                        resp = yield ("send", request, attempt_timeout)
                    except (openai.OpenAIError, httpx.HTTPError) as exc:
                        raise _classify(exc, entry.model, attempt_timeout) from exc
                    usage += _usage(resp)
                    result = _read(resp, entry, schema, aliases)
                except LLMError as err:
                    attempts.append(
                        _attempt(entry, type(err).__name__, t0, err.status_code)
                    )
                    errors.append(err)
                    logger.warning(
                        "LLM attempt failed: model=%s error=%s status=%s: %s",
                        entry.model, type(err).__name__, err.status_code, str(err)[:200],
                    )  # fmt: skip
                    if isinstance(err, _FATAL):
                        logger.error("LLM call stopped (no retry/fallback): %s", err)
                        raise
                    if isinstance(err, _TRANSIENT) and transient_left > 0:
                        transient_left -= 1
                        yield ("sleep", self._backoff(err))
                        continue
                    if isinstance(err, LLMInvalidOutput) and invalid_left > 0:
                        invalid_left -= 1
                        continue
                    break  # next model
                attempts.append(_attempt(entry, "ok", t0))
                result.usage = usage
                result.fallback_count = index
                result.attempts = attempts
                result.latency_ms = int((time.monotonic() - started) * 1000)
                logger.info(
                    "LLM call ok: model=%s fallbacks=%d attempts=%d tokens=%d+%d cost=%.6f latency_ms=%d",
                    result.model_used, index, len(attempts), usage.prompt_tokens,
                    usage.completion_tokens, usage.cost, result.latency_ms,
                )  # fmt: skip
                return result

        summary = "; ".join(f"{e.model}: {type(e).__name__}" for e in errors)
        logger.error("LLM call failed on every model: %s", summary)
        raise LLMAllModelsFailed(
            f"All models failed ({summary}). Last error: {errors[-1]}", errors=errors
        )

    def _body(
        self,
        feature: str | None,
        session_id: str | None,
        strict_routing: bool,
        needs: set[str],
    ) -> dict[str, Any]:
        cfg = self.config
        body: dict[str, Any] = {"usage": {"include": True}}
        if cfg.service:
            body["user"] = cfg.service
        if session_id:
            body["session_id"] = session_id
        trace = {
            k: v
            for k, v in (
                ("service", cfg.service),
                ("feature", feature),
                ("environment", cfg.environment),
            )
            if v
        }
        if trace:
            body["trace"] = trace
        if strict_routing:
            # Don't route to an endpoint that would ignore the schema or tools.
            body["provider"] = {"require_parameters": True}
        if "pdf" in needs:
            # Use the model's native PDF input, not OpenRouter's paid OCR fallback.
            body["plugins"] = [{"id": "file-parser", "pdf": {"engine": "native"}}]
        return body

    def _backoff(self, err: LLMError) -> float:
        wait = err.retry_after or 1.0 + random.random()
        return min(self.config.max_backoff_seconds, wait)


class _StreamAccumulator:
    """Builds a response shaped like a non-streamed one from stream chunks."""

    def __init__(self) -> None:
        self.id = self.model = self.provider = self.error = self.usage = None
        self.finish_reason: str | None = None
        self.text: list[str] = []
        self.tool_calls: dict[int, dict[str, str]] = {}
        self.reasoning_details: list[dict] = []

    def add(self, chunk: Any) -> None:
        self.id = self.id or getattr(chunk, "id", None)
        self.model = getattr(chunk, "model", None) or self.model
        self.provider = getattr(chunk, "provider", None) or self.provider
        self.error = getattr(chunk, "error", None) or self.error
        self.usage = getattr(chunk, "usage", None) or self.usage
        for choice in getattr(chunk, "choices", None) or []:
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    self.text.append(delta.content)
                for item in getattr(delta, "reasoning_details", None) or []:
                    self._add_reasoning(item)
                for call in delta.tool_calls or []:
                    slot = self.tool_calls.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    slot["id"] = call.id or slot["id"]
                    if call.function is not None:
                        slot["name"] = call.function.name or slot["name"]
                        slot["arguments"] += call.function.arguments or ""
            self.finish_reason = choice.finish_reason or self.finish_reason

    def _add_reasoning(self, item: Any) -> None:
        # Pieces of one reasoning block share an index: join their text/data so
        # the block is sent back on the next turn exactly as the model produced it.
        item = dict(item) if isinstance(item, dict) else item.model_dump()
        last = self.reasoning_details[-1] if self.reasoning_details else None
        same_block = (
            last is not None
            and item.get("index") is not None
            and last.get("index") == item.get("index")
            and last.get("type") == item.get("type")
        )
        if not same_block:
            self.reasoning_details.append(item)
            return
        for key in ("text", "summary", "data"):
            if item.get(key):
                last[key] = (last.get(key) or "") + item[key]
        for key in ("signature", "id", "format"):
            if item.get(key):
                last[key] = item[key]

    def result(self) -> SimpleNamespace:
        tool_calls = [
            SimpleNamespace(
                id=c["id"],
                function=SimpleNamespace(name=c["name"], arguments=c["arguments"]),
            )
            for _, c in sorted(self.tool_calls.items())
        ]
        message = SimpleNamespace(
            content="".join(self.text),
            tool_calls=tool_calls,
            reasoning_details=self.reasoning_details or None,
        )
        choices = []
        if self.text or tool_calls or self.finish_reason:
            choices = [
                SimpleNamespace(message=message, finish_reason=self.finish_reason)
            ]
        return SimpleNamespace(
            id=self.id,
            model=self.model,
            provider=self.provider,
            error=self.error,
            usage=self.usage,
            choices=choices,
        )


def _attempt(
    entry: ChainEntry, outcome: str, t0: float, status: int | None = None
) -> Attempt:
    return Attempt(entry.model, outcome, int((time.monotonic() - t0) * 1000), status)


def _usage(resp: Any) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    completion_details = getattr(u, "completion_tokens_details", None)
    prompt_details = getattr(u, "prompt_tokens_details", None)
    return Usage(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
        cost=float(getattr(u, "cost", 0) or 0),
    )


def _read(
    resp: Any,
    entry: ChainEntry,
    schema: type[BaseModel] | None,
    aliases: dict[str, str],
) -> LLMResult:
    error = getattr(resp, "error", None)
    if error or not getattr(resp, "choices", None):
        raise _from_error_body(error, entry.model)
    choice = resp.choices[0]
    text = (choice.message.content or "") if choice.message else ""
    finish = choice.finish_reason
    if finish == "content_filter":
        raise LLMModelRejected(
            f"{entry.model} refused (content filter)", model=entry.model
        )
    if finish == "error":
        raise LLMUnavailable(
            f"{entry.model} stopped with finish_reason=error", model=entry.model
        )
    parsed = None
    if schema is not None:
        if finish == "length":
            raise LLMInvalidOutput(
                f"{entry.model} hit max_tokens before finishing the JSON",
                model=entry.model,
            )
        parsed = parse_structured(text, schema, aliases, model_id=entry.model)
    tool_calls = _tool_calls(choice.message, entry.model)
    return LLMResult(
        text=text,
        parsed=parsed,
        tool_calls=tool_calls,
        citations=_citations(choice.message),
        message=_assistant_message(choice.message, text, tool_calls),
        model_used=getattr(resp, "model", None) or entry.model,
        finish_reason=finish,
        generation_id=getattr(resp, "id", None),
        provider=getattr(resp, "provider", None),
        raw=resp,
    )


def _tool_calls(message: Any, model: str) -> list[ToolCall]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            raise LLMInvalidOutput(
                f"{model} sent invalid JSON arguments for tool {call.function.name}",
                model=model,
            ) from e
        calls.append(ToolCall(call.id, call.function.name, arguments))
    return calls


def _citations(message: Any) -> list[Citation]:
    citations = []
    for annotation in getattr(message, "annotations", None) or []:
        cite = getattr(annotation, "url_citation", None)
        if getattr(annotation, "type", None) == "url_citation" and cite is not None:
            citations.append(
                Citation(
                    url=cite.url,
                    title=getattr(cite, "title", None),
                    content=getattr(cite, "content", None),
                )
            )
    return citations


def _assistant_message(
    message: Any, text: str, tool_calls: list[ToolCall]
) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in tool_calls
        ]
    # Gemini and Claude need their reasoning sent back on the next tool turn.
    reasoning_details = getattr(message, "reasoning_details", None)
    if reasoning_details:
        out["reasoning_details"] = reasoning_details
    return out


def _by_status(status: int | None, message: str, model: str) -> LLMError:
    if status == 401:
        return LLMAuthError(message, model=model, status_code=status)
    if status == 402:
        return LLMOutOfCredits(message, model=model, status_code=status)
    if status in (403, 404, 413) or (status == 400 and _MODEL_SPECIFIC.search(message)):
        return LLMModelRejected(message, model=model, status_code=status)
    if status == 408:
        return LLMTimeout(message, model=model, status_code=status)
    if status == 429:
        return LLMRateLimited(message, model=model, status_code=status)
    if status is None or status >= 500:
        return LLMUnavailable(message, model=model, status_code=status)
    return LLMBadRequest(message, model=model, status_code=status)


def _error_message(body: Any, fallback: str) -> str:
    err = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return fallback
    raw = (err.get("metadata") or {}).get("raw")
    msg = err.get("message") or fallback
    return f"{msg} ({str(raw)[:200]})" if raw else msg


def _classify(
    exc: openai.OpenAIError | httpx.HTTPError, model: str, timeout: float
) -> LLMError:
    # Errors while reading a stream come from httpx directly.
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return LLMTimeout(f"{model} timed out after {timeout:.0f}s", model=model)
    if isinstance(exc, (openai.APIConnectionError, httpx.HTTPError)):
        return LLMUnavailable(f"Connection error to OpenRouter: {exc}", model=model)
    if isinstance(exc, openai.APIStatusError):
        err = _by_status(exc.status_code, _error_message(exc.body, str(exc)), model)
        retry_after = exc.response.headers.get("retry-after", "")
        if retry_after.replace(".", "", 1).isdigit():
            err.retry_after = float(retry_after)
        return err
    if isinstance(exc, openai.APIError) and isinstance(exc.body, dict):
        # e.g. an error sent in the middle of a stream
        return _from_error_body(exc.body.get("error", exc.body), model)
    return LLMUnavailable(f"Unexpected OpenAI SDK error: {exc}", model=model)


def _from_error_body(error: Any, model: str) -> LLMError:
    """OpenRouter can answer HTTP 200 with an error object and no choices."""
    if isinstance(error, dict):
        code = str(error.get("code", ""))
        status = int(code) if code.isdigit() else 502
        message = _error_message({"error": error}, "Provider error")
        return _by_status(status, message, model)
    return LLMUnavailable(f"{model} returned no choices", model=model, status_code=200)
