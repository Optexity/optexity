from pydantic import BaseModel, Field, field_validator, model_validator

from optexity.schema.actions.llm_actions import LLMAction
from optexity.utils.utils import build_model


class LLMQueryAction(LLMAction):
    output_format: dict
    prompt_instructions: str
    output_variable_names: list[str] | None = None

    def build_model(self):
        return build_model(self.output_format)

    @field_validator("output_format")
    def validate_output_format(cls, v):
        if isinstance(v, dict):
            try:
                build_model(v)
            except Exception as e:
                raise ValueError(f"Invalid output_format dict: {e}")
            return v
        raise ValueError("output_format must be a dict")

    @model_validator(mode="after")
    def validate_output_var_in_format(self):
        if self.output_variable_names is not None:
            for key in self.output_variable_names:
                if key not in self.output_format:
                    raise ValueError(
                        f"Output variable {key} not found in output_format"
                    )
        return self

    def replace(self, pattern: str, replacement: str):
        self.prompt_instructions = self.prompt_instructions.replace(
            pattern, replacement
        )
        return self


class PythonScriptAction(BaseModel):
    execution_code: str


class SleepAction(BaseModel):
    sleep_time: float


class HumanInLoopAction(BaseModel):
    max_wait_time: float = Field(gt=0, le=600)


## State Jump Actions
class StateJumpAction(BaseModel):
    next_state_index: int


class FailStateAction(BaseModel):
    failure_message: str = "Automation completed at one of the failure states."

    def replace(self, pattern: str, replacement: str):
        self.failure_message = self.failure_message.replace(pattern, replacement)
        return self


class SetVariableAction(BaseModel):
    """Set a value in generated_variables.

    Use `value` for a static value, or `expression` for a computed value
    (evaluated after variable replacement, e.g. "{counter[0]} + 1").
    """

    name: str
    value: int | float | str | bool | None = None
    expression: str | None = None

    @model_validator(mode="after")
    def validate_one_provided(self):
        if self.value is None and self.expression is None:
            raise ValueError("Either 'value' or 'expression' must be provided")
        if self.value is not None and self.expression is not None:
            raise ValueError("Only one of 'value' or 'expression' can be provided")
        return self

    def replace(self, pattern: str, replacement: str):
        if self.expression:
            self.expression = self.expression.replace(pattern, replacement)
        return self


class MiscAction(BaseModel):
    """Container for miscellaneous actions (set_variable, llm_query, etc.).

    Exactly one sub-action must be provided.
    """

    set_variable: SetVariableAction | None = None
    llm_query: LLMQueryAction | None = None

    def replace(self, pattern: str, replacement: str):
        if self.set_variable:
            self.set_variable.replace(pattern, replacement)
        if self.llm_query:
            self.llm_query.replace(pattern, replacement)
        return self


# class RestartAction(StateJumpAction):
#     next_state_index: 0


# class StopAction(StateJumpAction):
#     next_state_index: -1
