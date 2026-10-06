"""Live smoke test: every tier model through the real OpenRouter API.

Not collected by pytest (no test_ prefix). Run manually:

    OPENROUTER_API_KEY=... python -m tests.models.openrouter.smoke_openrouter

For each model: one structured-output call (with a key containing a space), one
image call and one PDF call. Tagged service=test-adapter-smoke so the calls are
easy to find (and ignore) in the OpenRouter dashboard. Costs a few cents.
"""

import io
import os
import sys

from PIL import Image, ImageDraw
from pydantic import create_model

from optexity.inference.models.openrouter import (
    DEFAULT_TIERS,
    LLMConfig,
    LLMError,
    OpenRouterClient,
    user_message,
)

Extraction = create_model(
    "Extraction", **{"invoice total": (str, ...), "currency": (str, ...)}
)
Number = create_model("Number", value=(int, ...))


def _image_with_number(n: int) -> bytes:
    img = Image.new("RGB", (320, 120), "white")
    ImageDraw.Draw(img).text((20, 30), f"NUMBER: {n}", fill="black", font_size=48)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_with_number(n: int) -> bytes:
    img = Image.open(io.BytesIO(_image_with_number(n)))
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set")
        return 2

    failures = 0
    total_cost = 0.0
    for tier, models in DEFAULT_TIERS.items():
        for model in models:
            client = OpenRouterClient(
                LLMConfig.with_tier_overrides(
                    {tier: [model]},
                    api_key=key,
                    service="test-adapter-smoke",
                    environment="test",
                    invalid_output_retries=0,
                    transient_retries=0,
                )
            )
            checks = [
                (
                    "structured",
                    [user_message("Invoice: total due is 1,234.50 EUR.")],
                    Extraction,
                    lambda p: getattr(p, "invoice total")
                    .replace(",", "")
                    .startswith("1234.5"),
                ),
                (
                    "image",
                    [
                        user_message(
                            "What number is written in the image?",
                            images=[_image_with_number(73)],
                        )
                    ],
                    Number,
                    lambda p: p.value == 73,
                ),
                (
                    "pdf",
                    [
                        user_message(
                            "What number is written in the PDF?",
                            pdfs=[_pdf_with_number(58)],
                        )
                    ],
                    Number,
                    lambda p: p.value == 58,
                ),
            ]
            for name, messages, schema, check in checks:
                try:
                    r = client.complete(
                        messages, model=tier, schema=schema, feature=f"smoke-{name}"
                    )
                    ok = check(r.parsed)
                    total_cost += r.usage.cost
                    failures += 0 if ok else 1
                    print(
                        f"{'PASS' if ok else 'WRONG'}  {tier.value:8} {model:32} {name:10} "
                        f"-> {r.parsed.model_dump()} cost=${r.usage.cost:.5f} {r.latency_ms}ms"
                    )
                except LLMError as e:
                    failures += 1
                    print(
                        f"FAIL  {tier.value:8} {model:32} {name:10} -> {type(e).__name__}: {str(e)[:160]}"
                    )
    print(f"\n{failures} failure(s), total cost ${total_cost:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
