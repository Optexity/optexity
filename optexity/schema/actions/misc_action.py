from pydantic import BaseModel, Field, model_validator


class PythonScriptAction(BaseModel):
    execution_code: str


class SleepAction(BaseModel):
    sleep_time: float


class HumanInLoopAction(BaseModel):
    max_wait_time: float = Field(gt=0)


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
    """Container for miscellaneous actions (set_variable, etc.).

    Exactly one sub-action must be provided.
    """

    set_variable: SetVariableAction | None = None

    @model_validator(mode="after")
    def validate_one_provided(self):
        provided = {"set_variable": self.set_variable}
        non_null = [k for k, v in provided.items() if v is not None]
        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of set_variable must be provided in misc_action"
            )
        return self

    def replace(self, pattern: str, replacement: str):
        if self.set_variable:
            self.set_variable.replace(pattern, replacement)
        return self


# class RestartAction(StateJumpAction):
#     next_state_index: 0


# class StopAction(StateJumpAction):
#     next_state_index: -1
