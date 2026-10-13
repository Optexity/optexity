from enum import Enum


class ExitCodes(Enum):
    SUCCESS = 0
    AUTOMATION_FAILED = 10
    AUTOMATION_KILLED = 11
    WORKER_CRASHED = 12
    # Worker exited before the first node because the page did not answer a
    # locator query. The parent restarts the browser and reruns the task.
    BROWSER_UNRESPONSIVE = 13
