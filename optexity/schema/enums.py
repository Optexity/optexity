from enum import Enum


class ExitCodes(Enum):
    SUCCESS = 0
    AUTOMATION_FAILED = 10
    AUTOMATION_KILLED = 11
    WORKER_CRASHED = 12
