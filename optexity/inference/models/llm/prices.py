"""Estimated USD cost of a call from its token counts.

Prices come from LiteLLM's community price file (the same source litellm used),
refreshed in a background thread once a day. Billed spend is in Cloud Billing.
"""

import logging
import threading
import time

import httpx

from .config import LOCATION

logger = logging.getLogger(__name__)

_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_REFRESH_S = 24 * 3600
# Google and Anthropic charge 10% more on non-global endpoints.
_UPLIFT = 1.0 if LOCATION == "global" else 1.1

# USD per 1M tokens, global list price: input, output, cached input, cache write.
# Used until (or if) the price file can't be fetched.
_FALLBACK: dict[str, tuple[float, float, float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03, 0),
    "gemini-3.6-flash": (0.75, 3.75, 0.075, 0),
    "gemini-3.8-flash": (0.75, 3.75, 0.075, 0),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20, 0),
    "gemini-4-argon": (2.00, 10.00, 0.10, 0),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
}

_prices: dict[str, dict] = {}
_fetched_at = 0.0
_lock = threading.Lock()


def _fetch() -> None:
    global _prices
    try:
        data = httpx.get(_URL, timeout=10).json()
        _prices = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and "input_cost_per_token" in v
        }
    except Exception as e:
        logger.warning("Could not fetch LLM prices (%s); using built-in prices.", e)


def _refresh_in_background() -> None:
    global _fetched_at
    with _lock:
        if time.time() - _fetched_at < _REFRESH_S:
            return
        _fetched_at = time.time()
    threading.Thread(target=_fetch, daemon=True).start()


def _per_token(model: str) -> tuple[float, float, float, float] | None:
    _refresh_in_background()
    for key in (f"vertex_ai/{model}", model, f"gemini/{model}"):
        p = _prices.get(key)
        if p:
            inp = p["input_cost_per_token"]
            return (
                inp,
                p.get("output_cost_per_token") or 0,
                p.get("cache_read_input_token_cost") or inp,
                p.get("cache_creation_input_token_cost") or inp,
            )
    if model in _FALLBACK:
        return tuple(x / 1e6 for x in _FALLBACK[model])  # type: ignore[return-value]
    return None


def cost(
    model: str, prompt: int, cached: int, cache_write: int, output: int
) -> tuple[float, float]:
    """(input cost, output cost) in USD. `prompt` counts every input token,
    including cached and cache-write ones. shortcut: ignores >200k-token price
    tiers, add them if long prompts become common."""
    p = _per_token(model)
    if p is None:
        logger.warning("No price for model %s; cost reported as 0.", model)
        return 0.0, 0.0
    p_in, p_out, p_cached, p_write = p
    fresh = max(prompt - cached - cache_write, 0)
    input_cost = (fresh * p_in + cached * p_cached + cache_write * p_write) * _UPLIFT
    return input_cost, output * p_out * _UPLIFT
