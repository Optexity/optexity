"""browser-use ``BaseChatModel`` backed by litellm.

The browser-use agentic paths need a browser-use chat model, which is a
different type from the ``LLMModel`` the registry hands out. Rather than talk to
a provider SDK directly — which is how these paths came to run on a hardcoded
Gemini model, with their own key resolution and no fallback — this adapts
litellm to that interface so every LLM call in the engine goes through one
layer, on the model the task asked for.

It stays thin because litellm speaks the OpenAI wire format, so browser-use's
own ``OpenAIMessageSerializer`` does the message conversion.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

import litellm
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel

from optexity.utils.llm_settings import llm_settings, resolve_llm_api_key

from .litellm_model import litellm_fallbacks, reasoning_effort_for
from .llm_model import parse_json_from_completion

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ChatLiteLLM(BaseChatModel):
    """A litellm model string, exposed as a browser-use chat model.

    ``model`` is a full litellm model string ("provider/model"). Keeping the
    prefix matters twice over: litellm needs it to route, and browser-use's cost
    tracking looks the name up in litellm's own pricing table, which carries
    both the prefixed and bare forms.
    """

    model: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    # Escape hatch for per-provider request tuning (reasoning_effort, seed, ...).
    completion_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return self.model.split("/")[0] if "/" in self.model else "litellm"

    @property
    def name(self) -> str:
        return self.model

    def _usage(self, response: Any) -> ChatInvokeUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        return ChatInvokeUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            prompt_cached_tokens=(
                getattr(prompt_details, "cached_tokens", None)
                if prompt_details is not None
                else None
            ),
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            # Reasoning tokens are deliberately not added on top the way
            # browser-use's own ChatOpenAI does it: litellm already counts them
            # inside completion_tokens (same note in LLMModel.get_token_usage).
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    def _request(
        self, messages: list[BaseMessage], output_format: type[T] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": OpenAIMessageSerializer.serialize_messages(messages),
            "api_key": resolve_llm_api_key(self.model),
            # browser-use retries each step itself; letting litellm retry too
            # would multiply the attempts out.
            "num_retries": 0,
            "drop_params": True,
            "reasoning_effort": reasoning_effort_for(self.model),
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            body["max_tokens"] = self.max_output_tokens

        fallbacks = litellm_fallbacks(self.model)
        if fallbacks:
            # litellm only routes through its fallback path when this is present,
            # and that path hops to a worker thread — skip it when unconfigured.
            body["fallbacks"] = fallbacks

        if output_format is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "schema": SchemaOptimizer.create_optimized_json_schema(
                        output_format
                    ),
                    # Gemini rejects strict schemas, so unlike browser-use's
                    # ChatOpenAI this is not strict — the same shape LiteLLMModel
                    # already sends, with a lenient reparse below to cover it.
                    "strict": False,
                },
            }

        body.update(self.completion_kwargs)
        return body

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
            response = await litellm.acompletion(
                **self._request(messages, output_format)
            )
        except litellm.RateLimitError as e:
            raise ModelRateLimitError(message=str(e), model=self.name) from e
        except Exception as e:
            # browser-use's Agent treats ModelProviderError as a retryable step
            # failure; anything else escapes as a hard error.
            raise ModelProviderError(message=str(e), model=self.name) from e

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = self._usage(response)
        stop_reason = getattr(choice, "finish_reason", None)

        if output_format is None:
            return ChatInvokeCompletion(
                completion=content, usage=usage, stop_reason=stop_reason
            )

        if not content:
            raise ModelProviderError(
                message=f"{self.name} returned an empty structured-output response",
                model=self.name,
            )
        try:
            parsed = output_format.model_validate_json(content)
        except Exception:
            # drop_params=True means a provider that can't honour response_format
            # answers with prose- or fence-wrapped JSON rather than failing.
            try:
                parsed = parse_json_from_completion(content, output_format)
            except Exception as e:
                raise ModelProviderError(
                    message=f"Could not parse structured output from {self.name}: {e}",
                    model=self.name,
                ) from e
            logger.debug(
                f"{self.name} did not honour response_format; "
                f"recovered the schema from the raw completion."
            )
        return ChatInvokeCompletion(
            completion=parsed, usage=usage, stop_reason=stop_reason
        )


def build_agent_llm(model: str | None = None) -> ChatLiteLLM:
    """The chat model for the browser-use agentic paths.

    Callers pass the task's own model — ``normalize_model(task.llm_provider,
    task.llm_model_name)`` — so an agentic fallback runs on the same model as
    every other action in that task. ``None`` falls back to ``LLM_MODEL``, for
    the task-agnostic download-handling agent.
    """
    return ChatLiteLLM(model=model or llm_settings.LLM_MODEL)
