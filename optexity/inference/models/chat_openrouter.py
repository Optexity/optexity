"""browser-use ``BaseChatModel`` backed by the OpenRouter client, so the agentic
paths use the same models, fallback and usage tagging as every other LLM call.

OpenRouter speaks the OpenAI wire format, so browser-use's own
``OpenAIMessageSerializer`` does the message conversion.
"""

from dataclasses import dataclass, field
from typing import TypeVar, overload

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel

from .llm_model import get_client, get_config
from .openrouter import (
    LLMAllModelsFailed,
    LLMAuthError,
    LLMBadRequest,
    LLMConfigError,
    LLMError,
    LLMOutOfCredits,
    LLMRateLimited,
    Tier,
    Usage,
    resolve_chain,
)

T = TypeVar("T", bound=BaseModel)

# Errors that won't go away on a retry: they escape so the agent stops instead
# of burning its failure budget.
_FATAL = (LLMAuthError, LLMOutOfCredits, LLMBadRequest, LLMConfigError)


@dataclass
class ChatOpenRouter(BaseChatModel):
    """``target``: OpenRouter model ID, a `Tier`, or None for the default.

    ``model`` is a display string; browser-use reads it as a plain string.
    """

    target: str | Tier | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    # Usage-reporting tag sent with every call.
    feature: str = "agentic_task"
    # Token usage and cost of every successful call made through this instance.
    usage: Usage = field(default_factory=Usage)

    @property
    def model(self) -> str:
        if isinstance(self.target, Tier):
            return f"tier:{self.target.value}"
        return self.target or "default"

    @property
    def provider(self) -> str:
        return "openrouter"

    @property
    def name(self) -> str:
        return self.model

    @property
    def call_timeout(self) -> float:
        """Longest one call can take, including every retry and fallback, so the
        agent's own per-step timeout never cuts a call off mid-fallback."""
        config = get_config()
        attempts = config.transient_retries + 1
        chain = resolve_chain(config, self.target)
        return sum(
            e.timeout * attempts + config.max_backoff_seconds * config.transient_retries
            for e in chain
        )

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T]
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        try:
            result = await get_client().acomplete(
                OpenAIMessageSerializer.serialize_messages(messages),
                model=self.target,
                schema=output_format,
                # The agent's action schema is a large union; browser-use's own
                # optimized form, sent non-strict, is what providers accept.
                schema_override=(
                    SchemaOptimizer.create_optimized_json_schema(output_format)
                    if output_format is not None
                    else None
                ),
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                feature=self.feature,
            )
        except LLMAllModelsFailed as e:
            if all(isinstance(err, LLMRateLimited) for err in e.errors):
                raise ModelRateLimitError(message=str(e), model=self.name) from e
            raise ModelProviderError(message=str(e), model=self.name) from e
        except _FATAL:
            raise
        except LLMError as e:
            # browser-use treats ModelProviderError as a retryable step failure.
            raise ModelProviderError(message=str(e), model=self.name) from e

        self.usage += result.usage
        usage = ChatInvokeUsage(
            prompt_tokens=result.usage.prompt_tokens,
            prompt_cached_tokens=result.usage.cached_tokens or None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
        )
        completion = result.text if output_format is None else result.parsed
        return ChatInvokeCompletion(
            completion=completion, usage=usage, stop_reason=result.finish_reason
        )


def build_agent_llm(
    model: str | Tier | None = None, feature: str = "agentic_task"
) -> ChatOpenRouter:
    """The chat model for the browser-use agentic paths. Callers pass the task's
    own model so an agentic step runs on the same model as the rest of the task."""
    return ChatOpenRouter(target=model, feature=feature)
