import base64

import pytest
from pydantic import BaseModel, create_model

from optexity.inference.models.openrouter import (
    LLMBadRequest,
    LLMInvalidOutput,
    image_part,
    pdf_part,
    user_message,
)
from optexity.inference.models.openrouter.messages import (
    has_pdf,
    parse_structured,
    to_strict_schema,
)


class Item(BaseModel):
    name: str
    price: float | None = None


class Invoice(BaseModel):
    total: str = ""
    items: list[Item] = []


def test_strict_schema_requires_all_and_forbids_extra():
    schema, aliases = to_strict_schema(Invoice)
    assert aliases == {}
    assert schema["required"] == ["total", "items"]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["items"]["items"]  # $ref inlined
    assert item["required"] == ["name", "price"]
    assert item["additionalProperties"] is False
    assert "$defs" not in schema
    assert "default" not in str(schema)


def test_unsafe_keys_are_aliased_and_restored():
    Model = create_model("Extract", **{"invoice total": (str, ...), "ok": (bool, ...)})
    schema, aliases = to_strict_schema(Model)
    safe = [k for k in schema["properties"] if k != "ok"][0]
    assert aliases == {safe: "invoice total"}
    parsed = parse_structured(
        f'{{"{safe}": "12.50", "ok": true}}', Model, aliases, model_id="m"
    )
    assert getattr(parsed, "invoice total") == "12.50"


def test_lenient_parse_handles_fences_and_prose():
    text = 'Sure!\n```json\n{"total": "9", "items": []}\n```'
    assert parse_structured(text, Invoice, {}, model_id="m").total == "9"
    text = 'Here you go: {"total": "7", "items": [{"name": "a"}]} thanks'
    assert parse_structured(text, Invoice, {}, model_id="m").items[0].name == "a"


def test_unparseable_output_raises_invalid_output():
    with pytest.raises(LLMInvalidOutput):
        parse_structured("I cannot help with that", Invoice, {}, model_id="m")


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_png_bytes_and_base64_become_data_uris():
    for image in (PNG, base64.b64encode(PNG).decode()):
        part = image_part(image)
        assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_unknown_image_format_is_rejected():
    with pytest.raises(LLMBadRequest):
        image_part(b"not an image")


def test_pdf_part_and_message_inspection():
    msg = user_message("read this", images=[PNG], pdfs=[b"%PDF-1.4 x"])
    kinds = [p["type"] for p in msg["content"]]
    assert kinds == ["image_url", "file", "text"]
    assert has_pdf([msg])
    assert not has_pdf([user_message("text only", images=[PNG])])
    with pytest.raises(LLMBadRequest):
        pdf_part(b"not a pdf")


def test_lenient_parse_handles_escaped_quotes_and_braces_in_strings():
    text = 'Result: {"total": "say \\"hi\\" {not a brace}", "items": []} done'
    assert (
        parse_structured(text, Invoice, {}, model_id="m").total
        == 'say "hi" {not a brace}'
    )
