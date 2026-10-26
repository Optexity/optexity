"""Which models a call may use, in fallback order. The only place models are named."""

import functools
import logging
import re
from dataclasses import dataclass
from enum import Enum

from optexity.utils.llm_settings import llm_settings

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    LITE = "lite"
    STANDARD = "standard"
    PRO = "pro"


# Order inside a tier: Gemini, then Claude.
TIERS: dict[Tier, list[str]] = {
    Tier.LITE: ["gemini-3.5-flash-lite", "claude-haiku-5-5"],
    Tier.STANDARD: ["gemini-3.6-flash", "claude-sonnet-5-5"],
    Tier.PRO: ["gemini-3.1-pro-preview", "claude-opus-5-5"],
}
# Where a location lacks a tier model: gemini-3.1-pro-preview is served on global only.
LOCATION_TIERS: dict[str, dict[Tier, list[str]]] = {
    "us": {Tier.PRO: ["gemini-4-argon", "claude-opus-5-5"]},
}
DEFAULT_TIER = Tier.LITE
# Fallback for a pinned model that belongs to no tier (when LLM_MODEL_FALLBACK isn't set).
CATCH_ALL_TIER = Tier.STANDARD

# Gemini thinking level per tier; None leaves the model's default.
REASONING: dict[Tier, str | None] = {Tier.LITE: "medium", Tier.STANDARD: "medium", Tier.PRO: None}
# Per attempt, in seconds.
TIMEOUTS: dict[Tier, float] = {Tier.LITE: 120, Tier.STANDARD: 180, Tier.PRO: 300}

# Model names saved in workflows that Vertex doesn't serve, and what to use instead.
RETIRED: dict[str, str | Tier] = {
    "gemini-2.5-pro": Tier.PRO,
    "gemini-3.1-pro": Tier.PRO,
    "gemini-2.5-flash": Tier.STANDARD,
    "gemini-2.5-flash-lite": Tier.LITE,
    "claude-sonnet-4": "claude-sonnet-5-5",
    "claude-sonnet-4-5": "claude-sonnet-5-5",
    "claude-sonnet-4-6": "claude-sonnet-5-5",
    "claude-haiku-4-5": "claude-haiku-5-5",
    "claude-opus-4-7": "claude-opus-5-5",
}

LOCATION = llm_settings.GOOGLE_CLOUD_LOCATION
PROJECT = llm_settings.GOOGLE_CLOUD_PROJECT


@dataclass(frozen=True)
class Step:
    model: str
    tier: Tier | None


def tier_models(tier: Tier) -> list[str]:
    override = getattr(llm_settings, f"LLM_TIER_{tier.name}")
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return LOCATION_TIERS.get(LOCATION, {}).get(tier) or TIERS[tier]


def tier_of(model: str) -> Tier | None:
    """The tier a model belongs to in any location (gemini-4-argon is pro on global too)."""
    for tier in Tier:
        lists = [tier_models(tier), TIERS[tier], *(loc.get(tier, []) for loc in LOCATION_TIERS.values())]
        if any(model in models for models in lists):
            return tier
    return None


@functools.lru_cache(maxsize=256)  # also logs each warning once per string
def normalize(name: str) -> str | Tier | None:
    """A saved model string as a Vertex model ID (or a tier for retired models).

    Accepts the older forms found in workflows and env files: "gemini/x",
    "vertex_ai/x", "anthropic/claude-sonnet-4-6", "openrouter/anthropic/claude-sonnet-4.6",
    "bedrock/anthropic.claude-sonnet-4-20250514-v1:0". Models Vertex can't serve
    (e.g. OpenAI) give None, which runs the call on the default chain.
    """
    model = name.strip().lower().split("/")[-1]
    model = re.sub(r"^anthropic\.", "", model)
    model = re.sub(r"-v\d+(:\d+)?$", "", model)
    model = re.sub(r"[@-]\d{8}$", "", model)
    if model.startswith("claude"):
        model = model.replace(".", "-")
    if model in RETIRED:
        logger.warning("Model %r is not available on Vertex; using %s instead. Update the saved value.", name, RETIRED[model])
        return RETIRED[model]
    if model.startswith(("gemini", "claude")):
        return model
    logger.warning("Model %r is not available on Vertex; using the default models. Update the saved value.", name)
    return None


def _expand(target: str | Tier) -> list[str]:
    return tier_models(target) if isinstance(target, Tier) else [target]


def resolve_chain(model: str | Tier | None) -> list[Step]:
    """Models to try for one call, in order.

    - a Tier (hardcoded call site): that tier's models.
    - a model ID (action or task override): it, then LLM_MODEL_FALLBACK if set,
      else the rest of its tier (CATCH_ALL_TIER if it belongs to none).
    - None: LLM_MODEL if set (handled like a model ID), else the default tier's
      first model, then LLM_MODEL_FALLBACK if set, else the rest of the default tier.
    """
    fallback = normalize(llm_settings.LLM_MODEL_FALLBACK) if llm_settings.LLM_MODEL_FALLBACK else None
    rest = _expand(fallback) if fallback else None

    if isinstance(model, Tier):
        names, tier = tier_models(model), model
    else:
        target = normalize(model) if model else None
        if target is None and llm_settings.LLM_MODEL:
            target = normalize(llm_settings.LLM_MODEL)
        if target is None or isinstance(target, Tier):
            tier = target or DEFAULT_TIER
            models = tier_models(tier)
            names = [models[0], *rest] if rest else models
        else:
            tier = tier_of(target) or CATCH_ALL_TIER
            names = [target, *(rest if rest is not None else [m for m in tier_models(tier) if m != target])]

    chain: list[Step] = []
    for name in names:
        if name not in (s.model for s in chain):
            chain.append(Step(name, tier_of(name) or tier))
    return chain
