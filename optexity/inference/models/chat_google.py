"""ChatGoogle construction for the browser-use backed agentic paths.

These paths talk to browser-use's ``ChatGoogle`` directly rather than going
through ``get_llm_model``, so the Gemini 3.x request configuration lives here
instead of in the ``GeminiModels`` registry.
"""

from browser_use import ChatGoogle

# Model for the browser-use agentic paths (download handling and AgenticTask).
#
# Deliberately hardcoded rather than taken from ``task.llm_model_name``: a
# browser-use ``Agent`` needs a browser-use ``BaseChatModel``, which is a
# different type from the ``LLMModel`` that ``get_llm_model_with_fallback``
# returns, so the registry cannot feed these call sites. Only ChatGoogle is
# wired up here, so ``task.llm_provider`` cannot be honoured either — a task
# set to anthropic/openai still runs these two auxiliary agents on Gemini and
# needs GOOGLE_API_KEY. Keep this in step with the ``llm_model_name`` schema
# defaults; it is not derived from them.
AGENT_MODEL = "gemini-3.5-flash-lite"

# Gemini 3.x takes a discrete ``thinking_level`` in place of the integer
# ``thinking_budget``, and has no equivalent of ``thinking_budget=0``, so
# ``minimal`` is the floor.
#
# browser-use only auto-sets ``thinking_budget=0`` for model names containing
# "gemini-2.5-flash" or "gemini-flash" (llm/google/chat.py). A 3.x name matches
# neither, so it leaves ``thinking_config`` alone and the value below reaches
# the API untouched. Revisit if that gate is ever widened to cover 3.x names.
_AGENT_THINKING_LEVEL = "minimal"


def build_agent_llm(model: str = AGENT_MODEL) -> ChatGoogle:
    return ChatGoogle(
        model=model,
        # Gemini 3.x thinking models are documented to degrade when temperature
        # is lowered. ``None`` leaves the key unset so the API default applies
        # rather than browser-use's 0.5.
        temperature=None,
        config={"thinking_config": {"thinking_level": _AGENT_THINKING_LEVEL}},
    )
