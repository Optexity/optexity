"""Types shared by the router and the providers."""

import base64
from dataclasses import dataclass, field
from typing import Any, Literal

# rate_limit: try the next model now; transient: retry once, then next model;
# invalid_output: retry up to twice, then next model; rejected: this model can't
# serve the request, next model; fatal: stop (auth / config).
Kind = Literal["rate_limit", "transient", "invalid_output", "rejected", "fatal"]


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        kind: Kind,
        model: str | None = None,
        status: int | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.model = model
        self.status = status


def kind_for_status(status: int | None) -> Kind:
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "fatal"
    if status is None or status == 408 or status >= 500:
        return "transient"
    return "rejected"  # 400 (context too long, unsupported parameter), 404 (not served here)


@dataclass
class Request:
    """OpenAI-style messages: role system / user / assistant; content is a string
    or a list of {"type": "text"}, {"type": "image_url"} and {"type": "file"} parts
    carrying base64 data URLs."""

    messages: list[dict[str, Any]]
    schema: dict | None
    temperature: float | None
    max_tokens: int | None
    reasoning: str | None
    timeout: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0  # every input token, including cached / cache-write
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens
    thoughts_tokens: int = 0
    finish_reason: str | None = None


def decode_data_url(url: str) -> tuple[str, bytes]:
    """'data:image/png;base64,....' -> ('image/png', bytes)."""
    if not url.startswith("data:") or ";base64," not in url:
        raise LLMError(
            "Only base64 data URLs are supported for images and files", "fatal"
        )
    header, data = url[5:].split(";base64,", 1)
    return header, base64.b64decode(data)


def parts(content: str | list[dict] | None) -> list[dict]:
    """Message content as a list of parts, without empty text parts (both APIs reject them)."""
    items = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else content or []
    )
    return [p for p in items if p.get("type") != "text" or p.get("text")]
