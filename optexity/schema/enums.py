from enum import Enum

# The worker writes its crash error to the file this env var names; stderr only
# reaches the container log.
WORKER_ERROR_FILE_ENV = "WORKER_ERROR_FILE"


class ExitCodes(Enum):
    SUCCESS = 0
    AUTOMATION_FAILED = 10
    AUTOMATION_KILLED = 11
    WORKER_CRASHED = 12
    # Worker exited before the first node because the page did not answer a
    # locator query. The parent restarts the browser and reruns the task.
    BROWSER_UNRESPONSIVE = 13
