"""Errors, result types, model tiers, client config and fallback-chain resolution."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """``retryable``: whether sending the same request again could succeed."""

    retryable: bool = False

    def __init__(
        self, message: str, *, model: str | None = None, status_code: int | None = None
    ):
        super().__init__(message)
        self.model = model
        self.status_code = status_code
        self.retry_after: float | None = None


class LLMRateLimited(LLMError):
    retryable = True


class LLMUnavailable(LLMError):
    """5xx, connection errors, or HTTP 200 carrying an error instead of choices."""

    retryable = True


class LLMTimeout(LLMError):
    retryable = True


class LLMModelRejected(LLMError):
    """This model can't serve the request (context too long, moderation, no
    endpoint supporting the parameters); another model might."""


class LLMInvalidOutput(LLMError):
    """The model answered but its output didn't parse / validate."""


class LLMAuthError(LLMError):
    pass


class LLMOutOfCredits(LLMError):
    pass


class LLMBadRequest(LLMError):
    pass


class LLMConfigError(LLMError):
    """Invalid client configuration or call arguments."""


class LLMAllModelsFailed(LLMError):
    """Every model in the chain failed; ``errors`` has one entry per attempt."""

    def __init__(self, message: str, *, errors: list[LLMError]):
        last = errors[-1] if errors else None
        super().__init__(
            message,
            model=last.model if last else None,
            status_code=last.status_code if last else None,
        )
        self.errors = errors
        self.retryable = any(e.retryable for e in errors)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0  # already includes reasoning tokens
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0  # USD, as billed by OpenRouter

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.cached_tokens + other.cached_tokens,
            self.cost + other.cost,
        )


@dataclass
class Attempt:
    model: str
    outcome: str  # "ok" or the error class name
    latency_ms: int
    status_code: int | None = None


@dataclass
class WebSearch:
    """OpenRouter web search for a request. ``engine`` None lets OpenRouter pick
    (the model's native search where available)."""

    max_results: int = 5
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    engine: str | None = None

    def plugin(self) -> dict:
        plugin: dict = {"id": "web", "max_results": self.max_results}
        if self.include_domains:
            plugin["include_domains"] = self.include_domains
        if self.exclude_domains:
            plugin["exclude_domains"] = self.exclude_domains
        if self.engine:
            plugin["engine"] = self.engine
        return plugin


@dataclass
class Citation:
    url: str
    title: str | None = None
    content: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    text: str
    model_used: str
    usage: Usage = field(
        default_factory=Usage
    )  # all billed attempts, incl. failed ones
    parsed: BaseModel | None = None
    finish_reason: str | None = None
    generation_id: str | None = None
    provider: str | None = None
    fallback_count: int = 0
    attempts: list[Attempt] = field(default_factory=list)
    latency_ms: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)  # from web search
    # The assistant message as it should be appended to the conversation for the
    # next turn (tool calls and reasoning details included).
    message: dict | None = None
    raw: Any = None


class Tier(str, Enum):
    LITE = "lite"
    STANDARD = "standard"
    PRO = "pro"


DEFAULT_TIERS: dict[Tier, tuple[str, ...]] = {
    Tier.LITE: (
        "google/gemini-3.5-flash-lite",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.6-luna",
    ),
    Tier.STANDARD: (
        "google/gemini-3.6-flash",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.6-terra",
    ),
    Tier.PRO: (
        "google/gemini-2.5-pro",
        "anthropic/claude-opus-4.7",
        "openai/gpt-5.6-sol",
    ),
}

# None: the reasoning parameter isn't sent and the provider default applies.
DEFAULT_REASONING: dict[Tier, str | None] = {
    Tier.LITE: "medium",
    Tier.STANDARD: "medium",
    Tier.PRO: None,
}

# Per attempt, in seconds. A timeout is retried / falls back like any temporary error.
DEFAULT_TIMEOUTS: dict[Tier, float] = {
    Tier.LITE: 120.0,
    Tier.STANDARD: 180.0,
    Tier.PRO: 300.0,
}
_UNTIERED_TIMEOUT = DEFAULT_TIMEOUTS[Tier.STANDARD]


@dataclass(frozen=True)
class LLMConfig:
    """Client settings. Nothing is read from the environment here; the caller
    builds this from its own settings."""

    api_key: str | None
    base_url: str = "https://openrouter.ai/api/v1"
    service: str | None = (
        None  # sent as `user` / `trace.service` to split usage by service
    )
    environment: str | None = None
    default_model: str | None = None  # used when a call passes neither model nor tier
    env_fallback: str | None = (
        None  # single fallback for model/default calls, not tier calls
    )
    tiers: dict[Tier, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_TIERS)
    )
    transient_retries: int = 1  # same-model retries on 408/429/5xx/timeouts
    invalid_output_retries: int = 2  # same-model retries when output doesn't parse
    max_backoff_seconds: float = 10.0

    @classmethod
    def with_tier_overrides(
        cls, overrides: dict[Tier, list[str]] | None = None, **kwargs
    ) -> "LLMConfig":
        """Config with the given tier lists replacing the defaults."""
        tiers = dict(DEFAULT_TIERS)
        tiers.update({t: tuple(m) for t, m in (overrides or {}).items() if m})
        return cls(tiers=tiers, **kwargs)


def parse_model_list(value: str | None) -> list[str]:
    """Comma-separated model IDs (e.g. an LLM_TIER_PRO setting) -> list."""
    return [m.strip() for m in (value or "").split(",") if m.strip()]


@dataclass(frozen=True)
class ChainEntry:
    model: str
    reasoning: str | None
    timeout: float


# Old model strings the rules below can't convert -> OpenRouter ID.
LEGACY_MODEL_ALIASES: dict[str, str] = {}

# LiteLLM routes whose model part is converted by name.
_LEGACY_ROUTES = ("gemini/", "vertex_ai/", "bedrock/")
_logged_conversions: set[str] = set()


def normalize_model_id(model: str) -> str:
    """The OpenRouter ID for a model string.

    Accepts OpenRouter IDs (optionally prefixed "openrouter/") and older
    LiteLLM-style strings saved in workflows and env files:
    "gemini/gemini-2.5-pro" and bare "gemini-2.5-pro" -> "google/gemini-2.5-pro",
    "claude-opus-4-6" -> "anthropic/claude-opus-4.6",
    "bedrock/anthropic.claude-sonnet-4-20250514-v1:0" -> "anthropic/claude-sonnet-4".
    Anything else is passed through; OpenRouter rejects unknown IDs
    (LLMBadRequest).
    """
    original = model.strip()
    converted = LEGACY_MODEL_ALIASES.get(original) or _convert_legacy(
        original.removeprefix("openrouter/")
    )
    if (
        converted != original.removeprefix("openrouter/")
        and original not in _logged_conversions
    ):
        _logged_conversions.add(original)
        logger.warning(
            "Converted model %r to OpenRouter ID %r; update the saved value.",
            original,
            converted,
        )
    return converted


def _convert_legacy(model: str) -> str:
    if model.startswith(_LEGACY_ROUTES) or "/" not in model:
        name = model.split("/", 1)[1] if "/" in model else model
        name = re.sub(r"^anthropic\.", "", name)  # bedrock: anthropic.claude-...
        name = re.sub(r"-v\d+(:\d+)?$", "", name)  # bedrock: ...-v1:0
        vendor = _vendor_of(name)
        if vendor is None:
            return model
        model = f"{vendor}/{name}"
    vendor, name = model.split("/", 1)
    if vendor == "anthropic":
        name = _anthropic_name(name)
    return f"{vendor}/{name}"


def _vendor_of(name: str) -> str | None:
    if name.startswith("gemini"):
        return "google"
    if name.startswith("claude"):
        return "anthropic"
    if re.match(r"^(gpt-|chatgpt-|o\d)", name):
        return "openai"
    return None


def _anthropic_name(name: str) -> str:
    # Anthropic's own IDs use dashes and dates; OpenRouter's use dots, no date.
    name = re.sub(r"(@\d{8}|-\d{8})$", "", name)
    name = re.sub(r"^(claude-(?:opus|sonnet|haiku)-\d+)-(\d+)$", r"\1.\2", name)
    return re.sub(r"^claude-(\d+)-(\d+)-(opus|sonnet|haiku)$", r"claude-\1.\2-\3", name)


def tier_of(config: LLMConfig, model: str) -> Tier | None:
    return next((t for t, models in config.tiers.items() if model in models), None)


def _entry(config: LLMConfig, model: str, tier: Tier | None = None) -> ChainEntry:
    tier = tier or tier_of(config, model)
    reasoning = DEFAULT_REASONING[tier] if tier else None
    timeout = DEFAULT_TIMEOUTS[tier] if tier else _UNTIERED_TIMEOUT
    return ChainEntry(model, reasoning, timeout)


def _rest_of_tier(config: LLMConfig, model: str) -> list[str]:
    tier = tier_of(config, model)
    if tier is None:
        return []
    models = list(config.tiers[tier])
    return models[models.index(model) + 1 :]


def resolve_chain(
    config: LLMConfig,
    model: str | Tier | None = None,
) -> list[ChainEntry]:
    """Models to try for one call, in order.

    - a model ID: that model, then ``env_fallback`` if set, else the rest of the
      model's tier (nothing if it isn't in a tier).
    - a ``Tier``: the tier's models; ``env_fallback`` is ignored.
    - ``None``: ``default_model`` (else the lite primary), then ``env_fallback``
      if set, else the lite tier.

    A model that can't serve a request (unsupported parameters, context too
    long) is skipped at call time from OpenRouter's error.
    """
    fallback = normalize_model_id(config.env_fallback) if config.env_fallback else None

    if isinstance(model, Tier):
        entries = [_entry(config, m, model) for m in config.tiers[model]]
    elif model:
        chosen = normalize_model_id(model)
        rest = [fallback] if fallback else _rest_of_tier(config, chosen)
        entries = [_entry(config, m) for m in [chosen, *rest]]
    elif config.default_model:
        chosen = normalize_model_id(config.default_model)
        rest = [fallback] if fallback else list(config.tiers[Tier.LITE])
        entries = [_entry(config, m) for m in [chosen, *rest]]
    else:
        lite = config.tiers[Tier.LITE]
        entries = [
            _entry(config, m) for m in ([lite[0], fallback] if fallback else lite)
        ]

    chain: list[ChainEntry] = []
    for e in entries:
        if not any(c.model == e.model for c in chain):
            chain.append(e)
    if not chain:
        raise LLMConfigError("No models configured for this call")
    return chain
