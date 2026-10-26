"""browser-use ``BaseChatModel`` backed by ``optexity.inference.models.llm``.

The browser-use agentic paths need a browser-use chat model, which is a
different type from the ``LLMModel`` the registry hands out. This adapts the LLM
layer to that interface so agentic steps get the same models, fallback, retries
and logging as every other LLM call. browser-use's ``OpenAIMessageSerializer``
produces the OpenAI-style messages the layer takes.
"""

import logging
from dataclasses import dataclass
from typing import TypeVar, overload

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel

from .llm import LLMError, acomplete
from .llm_model import parse_json_from_completion

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ChatLLM(BaseChatModel):
    """``target`` is a model ID or None for the configured default; ``model`` is
    the display name browser-use reads."""

    target: str | None = None
    feature: str = "agentic_task"
    temperature: float | None = None
    max_output_tokens: int | None = None

    @property
    def model(self) -> str:
        return self.target or "default"

    @property
    def provider(self) -> str:
        return "vertex"

    @property
    def name(self) -> str:
        return self.model

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

        def parse(content: str) -> T:
            if not content:
                raise ValueError("empty structured-output response")
            try:
                return output_format.model_validate_json(content)
            except Exception:
                # Prose- or fence-wrapped JSON (e.g. when a schema can't be sent as-is).
                return parse_json_from_completion(content, output_format)

        try:
            result = await acomplete(
                OpenAIMessageSerializer.serialize_messages(messages),
                model=self.target,
                schema=(
                    SchemaOptimizer.create_optimized_json_schema(output_format)
                    if output_format is not None
                    else None
                ),
                parse=parse if output_format is not None else None,
                feature=self.feature,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
        except LLMError as e:
            # browser-use's Agent treats ModelProviderError as a retryable step
            # failure; anything else escapes as a hard error.
            if e.kind == "rate_limit":
                raise ModelRateLimitError(message=str(e), model=self.name) from e
            raise ModelProviderError(message=str(e), model=self.name) from e

        usage = ChatInvokeUsage(
            prompt_tokens=result.usage.input_tokens,
            prompt_cached_tokens=None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        )
        return ChatInvokeCompletion(
            completion=result.parsed if output_format is not None else result.text,
            usage=usage,
            stop_reason=result.finish_reason,
        )


def build_agent_llm(model: str | None = None, feature: str = "agentic_task") -> ChatLLM:
    """The chat model for the browser-use agentic paths.

    Callers pass the task's own model — ``normalize_model(task.llm_provider,
    task.llm_model_name)`` — so an agentic fallback runs on the same model as
    every other action in that task. ``None`` uses the configured default, for
    the task-agnostic download-handling agent.
    """
    return ChatLLM(target=model, feature=feature)
