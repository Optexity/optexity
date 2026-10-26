"""Runs one LLM call across its chain of models with retries and fallback.

Per model: transient errors (5xx, timeouts) are retried once, unparseable output
up to twice, then the next model is tried; a rate-limited model is skipped for a
short while; auth / config errors stop the call. Every attempt is logged.
"""

import asyncio
import random
import re
import time
import uuid
from collections.abc import Callable, Generator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from optexity.schema.token_usage import TokenUsage
from optexity.utils.llm_settings import llm_settings

from . import claude, gemini
from .config import LOCATION, REASONING, TIMEOUTS, Step, Tier, resolve_chain
from .log import emit
from .prices import cost
from .types import LLMError, Reply, Request

# Set once per task run so every LLM log entry carries the task id.
TASK_ID: ContextVar[str | None] = ContextVar("llm_task_id", default=None)

_COOLDOWN_S = 45.0
_cooldown_until: dict[str, float] = {}


@dataclass
class LLMResult:
    text: str
    parsed: Any
    model: str
    usage: TokenUsage  # every billed attempt, including failed ones
    finish_reason: str | None


def _provider(model: str):
    return claude if model.startswith("claude") else gemini


def _label(value: str) -> str:
    # Vertex labels: lowercase letters, digits, "_" and "-", at most 63 characters.
    return re.sub(r"[^a-z0-9_-]", "_", value.lower())[:63]


def _usage(model: str, reply: Reply) -> TokenUsage:
    input_cost, output_cost = cost(
        model,
        reply.prompt_tokens,
        reply.cached_tokens,
        reply.cache_write_tokens,
        reply.output_tokens,
    )
    total = reply.prompt_tokens + reply.output_tokens
    return TokenUsage(
        input_tokens=reply.prompt_tokens,
        output_tokens=reply.output_tokens,
        thoughts_tokens=reply.thoughts_tokens,
        total_tokens=total,
        calculated_total_tokens=total + reply.thoughts_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


def _available(chain: list[Step]) -> list[Step]:
    now = time.monotonic()
    ready = [s for s in chain if _cooldown_until.get(s.model, 0) <= now]
    return ready or chain[-1:]


def _run(
    messages: list[dict],
    model: str | Tier | None,
    schema: dict | None,
    parse: Callable[[str], Any] | None,
    feature: str,
    temperature: float | None,
    max_tokens: int | None,
) -> Generator[tuple, Any, LLMResult]:
    """Yields ("call", provider, model, request) or ("sleep", seconds); the driver
    sends back the provider's Reply or throws its exception."""
    chain = _available(resolve_chain(model))
    requested = model.value if isinstance(model, Tier) else (model or "default")
    call_id = uuid.uuid4().hex[:16]
    total = TokenUsage()
    errors: list[LLMError] = []
    attempt = 0

    for index, step in enumerate(chain):
        provider = _provider(step.model)
        tier = step.tier
        request = Request(
            messages=messages,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=REASONING[tier] if tier else None,
            timeout=TIMEOUTS[tier] if tier else TIMEOUTS[Tier.STANDARD],
            labels={
                "service": _label(llm_settings.LLM_SERVICE_NAME),
                "feature": _label(feature),
                "environment": _label(llm_settings.LLM_ENVIRONMENT),
                "tier": tier.value if tier else "none",
            },
        )
        transient_left, invalid_left = 1, 2
        while True:
            attempt += 1
            started = time.monotonic()
            reply: Reply | None = None
            try:
                try:
                    reply = yield ("call", provider, step.model, request)
                except LLMError:
                    raise
                except Exception as exc:
                    raise provider.classify(exc, step.model) from exc
                try:
                    parsed = parse(reply.text) if parse else None
                except Exception as exc:
                    raise LLMError(
                        f"{step.model} output did not match the schema ({type(exc).__name__})",
                        "invalid_output",
                        step.model,
                    ) from exc
            except LLMError as err:
                errors.append(err)
                if err.kind == "rate_limit":
                    _cooldown_until[step.model] = time.monotonic() + _COOLDOWN_S
                retry = (err.kind == "transient" and transient_left > 0) or (
                    err.kind == "invalid_output" and invalid_left > 0
                )
                final = err.kind == "fatal" or (not retry and index == len(chain) - 1)
                usage = _usage(step.model, reply) if reply else TokenUsage()
                total += usage
                _log(
                    call_id,
                    attempt,
                    step,
                    chain,
                    requested,
                    feature,
                    started,
                    usage,
                    reply.cached_tokens if reply else 0,
                    err,
                    final,
                    errors,
                )
                if err.kind == "fatal":
                    raise err
                if retry:
                    if err.kind == "transient":
                        transient_left -= 1
                        yield ("sleep", 1 + random.random())
                    else:
                        invalid_left -= 1
                    continue
                break  # next model

            usage = _usage(step.model, reply)
            total += usage
            _log(
                call_id,
                attempt,
                step,
                chain,
                requested,
                feature,
                started,
                usage,
                reply.cached_tokens,
                None,
                True,
                errors,
            )
            return LLMResult(reply.text, parsed, step.model, total, reply.finish_reason)

    first = errors[0]
    kind = "rate_limit" if all(e.kind == "rate_limit" for e in errors) else first.kind
    summary = "; ".join(f"{e.model}: {e.kind}" for e in errors)
    raise LLMError(
        f"All models failed ({summary}). First error: {first}",
        kind,
        first.model,
        first.status,
    )


def _log(
    call_id,
    attempt,
    step,
    chain,
    requested,
    feature,
    started,
    usage,
    cached,
    err,
    final,
    errors,
) -> None:
    fallback = step.model != chain[0].model
    emit(
        {
            "service": llm_settings.LLM_SERVICE_NAME,
            "feature": feature,
            "environment": llm_settings.LLM_ENVIRONMENT,
            "location": LOCATION,
            "task_id": TASK_ID.get(),
            "call_id": call_id,
            "requested": requested,
            "tier": step.tier.value if step.tier else None,
            "model_used": step.model,
            "provider": "claude" if step.model.startswith("claude") else "gemini",
            "attempt": attempt,
            "final": final,
            "fallback": fallback,
            "fallback_reason": (
                next((e.kind for e in errors if e.model == chain[0].model), None)
                if fallback
                else None
            ),
            "status": "error" if err else "ok",
            "error_class": err.kind if err else None,
            "error_status": err.status if err else None,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": cached,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(usage.total_cost, 8),
        }
    )


def complete(
    messages: list[dict],
    *,
    model: str | Tier | None = None,
    schema: dict | None = None,
    parse: Callable[[str], Any] | None = None,
    feature: str = "unknown",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """Send `messages` (OpenAI-style) to `model`: a Tier, a model ID or None for
    the default. `schema` requests JSON output; `parse` turns the text into the
    result (its exceptions count as invalid output and are retried)."""
    run = _run(messages, model, schema, parse, feature, temperature, max_tokens)
    step = next(run)
    while True:
        try:
            if step[0] == "sleep":
                time.sleep(step[1])
                step = run.send(None)
                continue
            _, provider, name, request = step
            try:
                reply = provider.call(request, name)
            except Exception as exc:
                step = run.throw(exc)
            else:
                step = run.send(reply)
        except StopIteration as done:
            return done.value


async def acomplete(
    messages: list[dict],
    *,
    model: str | Tier | None = None,
    schema: dict | None = None,
    parse: Callable[[str], Any] | None = None,
    feature: str = "unknown",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """Async version of `complete`."""
    run = _run(messages, model, schema, parse, feature, temperature, max_tokens)
    step = next(run)
    while True:
        try:
            if step[0] == "sleep":
                await asyncio.sleep(step[1])
                step = run.send(None)
                continue
            _, provider, name, request = step
            try:
                reply = await provider.acall(request, name)
            except Exception as exc:
                step = run.throw(exc)
            else:
                step = run.send(reply)
        except StopIteration as done:
            return done.value
