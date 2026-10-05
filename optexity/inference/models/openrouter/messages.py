"""Message parts for images and PDFs, strict JSON schemas, and parsing model output."""

import ast
import base64
import copy
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .core import LLMBadRequest, LLMInvalidOutput

_SAFE_KEY = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ALIAS_PREFIX = "k__"
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def image_part(image: bytes | str) -> dict[str, Any]:
    """PNG / JPEG / WEBP image (raw bytes or base64 string) -> content part."""
    raw = base64.b64decode(image) if isinstance(image, str) else image
    mime = next((m for sig, m in _IMAGE_SIGNATURES if raw.startswith(sig)), None)
    if mime is None and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    if mime is None:
        raise LLMBadRequest("Unsupported image format (expected PNG, JPEG or WEBP)")
    b64 = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def pdf_part(pdf: bytes, filename: str = "document.pdf") -> dict[str, Any]:
    if not pdf.startswith(b"%PDF"):
        raise LLMBadRequest("PDF bytes don't start with %PDF")
    b64 = base64.b64encode(pdf).decode("ascii")
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "file_data": f"data:application/pdf;base64,{b64}",
        },
    }


def user_message(
    text: str, *, images: list[bytes | str] | tuple = (), pdfs: list[bytes] | tuple = ()
) -> dict[str, Any]:
    """A user message with optional images / PDFs placed before the text."""
    content = [image_part(i) for i in images]
    content += [pdf_part(p, f"document_{n + 1}.pdf") for n, p in enumerate(pdfs)]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def inspect_messages(messages: list[dict[str, Any]]) -> tuple[set[str], int]:
    """(capabilities the messages need, rough token estimate of their text)."""
    needs: set[str] = set()
    chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for part in content or []:
            kind = part.get("type")
            if kind == "image_url":
                needs.add("vision")
            elif kind == "file":
                needs.add("pdf")
            elif kind == "text":
                chars += len(part.get("text", ""))
    return needs, chars // 4


def to_strict_schema(model: type[BaseModel]) -> tuple[dict, dict[str, str]]:
    """Strict JSON schema for ``response_format``: every property required,
    ``additionalProperties: false``, ``$ref`` inlined, ``default`` removed.

    Property names providers reject (spaces, symbols, >64 chars) are replaced
    by aliases; returns (schema, {alias: original name}).
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    aliases: dict[str, str] = {}
    reverse: dict[str, str] = {}

    def alias_for(key: str) -> str:
        if _SAFE_KEY.match(key) and not key.startswith(_ALIAS_PREFIX):
            return key
        if key not in reverse:
            reverse[key] = f"{_ALIAS_PREFIX}{len(reverse)}"
            aliases[reverse[key]] = key
        return reverse[key]

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 64:
            raise ValueError("Schema too deeply nested (recursive model?)")
        if isinstance(node, list):
            return [walk(n, depth + 1) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
            target.update({k: v for k, v in node.items() if k != "$ref"})
            return walk(target, depth + 1)
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "default":
                continue
            if k == "properties" and isinstance(v, dict):
                out[k] = {alias_for(pk): walk(pv, depth + 1) for pk, pv in v.items()}
            else:
                out[k] = walk(v, depth + 1)
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return walk(schema), aliases


def parse_structured(
    text: str, model: type[BaseModel], aliases: dict[str, str], *, model_id: str
) -> BaseModel:
    try:
        return model.model_validate(_restore_keys(_extract_json(text), aliases))
    except (ValueError, ValidationError) as e:
        raise LLMInvalidOutput(
            f"Output from {model_id} did not match {model.__name__}: {str(e)[:300]}",
            model=model_id,
        ) from e


def _restore_keys(data: Any, aliases: dict[str, str]) -> Any:
    if not aliases:
        return data
    if isinstance(data, dict):
        return {aliases.get(k, k): _restore_keys(v, aliases) for k, v in data.items()}
    if isinstance(data, list):
        return [_restore_keys(v, aliases) for v in data]
    return data


def _extract_json(text: str) -> Any:
    """Plain JSON, else a ```json fence or the first {...} block in the text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    block = _first_json_object(text)
    if block:
        candidates.append(block)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
    raise ValueError("no JSON object found in the model output")


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
