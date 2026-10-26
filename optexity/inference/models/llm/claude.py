"""Claude on Vertex AI (anthropic[vertex]). The only module that imports anthropic."""

import base64
import copy
import json

import anthropic
from anthropic import AnthropicVertex, AsyncAnthropicVertex

from .config import LOCATION, PROJECT
from .types import LLMError, Reply, Request, decode_data_url, kind_for_status, parts

_DEFAULT_MAX_TOKENS = 16384  # Claude requires max_tokens
_STRUCTURED_BETA = {"anthropic-beta": "structured-outputs-2025-11-13"}
_sync: AnthropicVertex | None = None
_async: AsyncAnthropicVertex | None = None


def _sync_client() -> AnthropicVertex:
    global _sync
    if _sync is None:
        _sync = AnthropicVertex(project_id=PROJECT, region=LOCATION)
    return _sync


def _async_client() -> AsyncAnthropicVertex:
    global _async
    if _async is None:
        _async = AsyncAnthropicVertex(project_id=PROJECT, region=LOCATION)
    return _async


def _closed(schema: dict) -> dict | None:
    """The schema with every object closed (Claude requires additionalProperties:
    false), or None if it has free-form maps, which Claude's JSON mode rejects."""
    schema = copy.deepcopy(schema)
    nodes = [schema]
    while nodes:
        node = nodes.pop()
        if isinstance(node, list):
            nodes.extend(node)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                if node.get("additionalProperties") not in (None, False) or "properties" not in node:
                    return None
                node["additionalProperties"] = False
            nodes.extend(node.values())
    return schema


def _block(p: dict) -> dict:
    if p["type"] == "text":
        return {"type": "text", "text": p["text"]}
    if p["type"] in ("image_url", "file"):
        url = p["image_url"]["url"] if p["type"] == "image_url" else p["file"]["file_data"]
        mime, data = decode_data_url(url)
        kind = "document" if mime == "application/pdf" else "image"
        return {"type": kind, "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(data).decode()}}
    return {"type": "text", "text": p.get("refusal", "")}


def _payload(req: Request, model: str) -> dict:
    system = [p["text"] for m in req.messages if m["role"] == "system" for p in parts(m["content"])]
    messages = [
        {"role": m["role"], "content": [_block(p) for p in parts(m["content"])]}
        for m in req.messages
        if m["role"] != "system" and parts(m["content"])
    ]
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": req.max_tokens or _DEFAULT_MAX_TOKENS,
        "timeout": req.timeout,
        # Billing labels go in this header for partner models (base64 JSON).
        "extra_headers": {"X-Vertex-AI-Labels": base64.b64encode(json.dumps(req.labels).encode()).decode()},
    }
    # temperature and seed are rejected by Claude 5.5 models, so they are not sent.
    if req.schema is not None:
        closed = _closed(req.schema)
        if closed is not None:
            kwargs["extra_body"] = {"output_format": {"type": "json_schema", "schema": closed}}
            kwargs["extra_headers"] |= _STRUCTURED_BETA
        else:
            system.append(f"Respond only with JSON matching this JSON schema:\n{json.dumps(req.schema)}")
    if system:
        kwargs["system"] = "\n\n".join(system)
    return kwargs


def _read(msg: anthropic.types.Message, model: str) -> Reply:
    if msg.stop_reason == "refusal":
        raise LLMError(f"{model} refused the request", "rejected", model)
    u = msg.usage
    cached = u.cache_read_input_tokens or 0
    written = u.cache_creation_input_tokens or 0
    return Reply(
        text="".join(b.text for b in msg.content if b.type == "text"),
        prompt_tokens=u.input_tokens + cached + written,
        cached_tokens=cached,
        cache_write_tokens=written,
        output_tokens=u.output_tokens,
        finish_reason=msg.stop_reason,
    )


def call(req: Request, model: str) -> Reply:
    return _read(_sync_client().messages.create(**_payload(req, model)), model)


async def acall(req: Request, model: str) -> Reply:
    return _read(await _async_client().messages.create(**_payload(req, model)), model)


def classify(exc: Exception, model: str) -> LLMError:
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(f"{model}: {exc.status_code} {exc.message}", kind_for_status(exc.status_code), model, exc.status_code)
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError, TimeoutError)):
        return LLMError(f"{model}: {type(exc).__name__}", "transient", model)
    return LLMError(f"{model}: {type(exc).__name__}: {exc}", "fatal", model)
