"""Gemini on Vertex AI (google-genai). The only module that imports google.genai."""

import httpx
from google import genai
from google.genai import errors, types

from .config import LOCATION, PROJECT
from .types import LLMError, Reply, Request, decode_data_url, kind_for_status, parts

_client: genai.Client | None = None  # kept referenced: an unreferenced client gets closed
_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    return _client


def _part(p: dict) -> types.Part:
    if p["type"] == "text":
        return types.Part.from_text(text=p["text"])
    if p["type"] == "image_url":
        mime, data = decode_data_url(p["image_url"]["url"])
        return types.Part.from_bytes(data=data, mime_type=mime)
    if p["type"] == "file":
        mime, data = decode_data_url(p["file"]["file_data"])
        return types.Part.from_bytes(data=data, mime_type=mime)
    return types.Part.from_text(text=p.get("refusal", ""))


def _payload(req: Request, model: str) -> dict:
    system = [p["text"] for m in req.messages if m["role"] == "system" for p in parts(m["content"])]
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[_part(p) for p in parts(m["content"])])
        for m in req.messages
        if m["role"] != "system" and parts(m["content"])
    ]
    config = types.GenerateContentConfig(
        system_instruction="\n\n".join(system) or None,
        temperature=req.temperature,
        max_output_tokens=req.max_tokens,
        labels=req.labels,
        http_options=types.HttpOptions(timeout=int(req.timeout * 1000)),
    )
    if req.schema is not None:
        config.response_mime_type = "application/json"
        config.response_json_schema = req.schema
    # Gemini 3+ takes a thinking level; "minimal" and token budgets are rejected by some models.
    if req.reasoning and model.startswith(("gemini-3", "gemini-4")):
        config.thinking_config = types.ThinkingConfig(thinking_level=req.reasoning)
    return {"model": model, "contents": contents, "config": config}


def _read(resp: types.GenerateContentResponse, model: str) -> Reply:
    feedback = resp.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise LLMError(f"{model} blocked the prompt: {feedback.block_reason}", "rejected", model)
    candidate = resp.candidates[0] if resp.candidates else None
    finish = candidate.finish_reason.name if candidate and candidate.finish_reason else None
    if candidate is None or finish in _BLOCKED:
        raise LLMError(f"{model} returned no answer (finish_reason={finish})", "rejected", model)
    text = "".join(p.text for p in (candidate.content.parts if candidate.content else []) or [] if p.text and not p.thought)
    u = resp.usage_metadata
    thoughts = (u.thoughts_token_count or 0) if u else 0
    return Reply(
        text=text,
        prompt_tokens=(u.prompt_token_count or 0) if u else 0,
        cached_tokens=(u.cached_content_token_count or 0) if u else 0,
        output_tokens=((u.candidates_token_count or 0) if u else 0) + thoughts,
        thoughts_tokens=thoughts,
        finish_reason=finish,
    )


def call(req: Request, model: str) -> Reply:
    return _read(_get_client().models.generate_content(**_payload(req, model)), model)


async def acall(req: Request, model: str) -> Reply:
    return _read(await _get_client().aio.models.generate_content(**_payload(req, model)), model)


def classify(exc: Exception, model: str) -> LLMError:
    if isinstance(exc, errors.APIError):
        return LLMError(f"{model}: {exc.code} {exc.message}", kind_for_status(exc.code), model, exc.code)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError)):
        return LLMError(f"{model}: {type(exc).__name__}", "transient", model)
    return LLMError(f"{model}: {type(exc).__name__}: {exc}", "fatal", model)
