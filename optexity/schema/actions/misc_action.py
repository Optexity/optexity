from pydantic import BaseModel


class PythonScriptAction(BaseModel):
    execution_code: str


class SleepAction(BaseModel):
    sleep_time: float


class HumanInLoopAction(BaseModel):
    max_wait_time: float


## State Jump Actions
class StateJumpAction(BaseModel):
    next_state_index: int


# class RestartAction(StateJumpAction):
#     next_state_index: 0


# class StopAction(StateJumpAction):
#     next_state_index: -1
