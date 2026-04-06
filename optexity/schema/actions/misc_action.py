from pydantic import BaseModel


class PythonScriptAction(BaseModel):
    execution_code: str


class SleepAction(BaseModel):
    sleep_time: float


class FailStateAction(BaseModel):
    failure_message: str = "Automation completed at one of the failure states."

    def replace(self, pattern: str, replacement: str):
        self.failure_message = self.failure_message.replace(pattern, replacement)
        return self


## State Jump Actions
class StateJumpAction(BaseModel):
    next_state_index: int


# class RestartAction(StateJumpAction):
#     next_state_index: 0


# class StopAction(StateJumpAction):
#     next_state_index: -1
