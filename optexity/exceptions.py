class AssertLocatorPresenceException(Exception):
    def __init__(self, message: str, command: str, original_error: Exception):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.command = command


class ElementNotFoundInAxtreeException(Exception):
    def __init__(self, message: str, command: str, original_error: Exception):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.command = command


class AxtreeIndexActionFailedException(Exception):
    def __init__(self, message: str, index: int, original_error):
        super().__init__(message)
        self.message = message
        self.index = index
        self.original_error = original_error


class HumanInLoopTimeoutException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ExpectedDownloadFailedException(Exception):
    """Raised when a node has expect_download=True but the action did not
    produce a downloaded file. This fails the task with a fixed message."""

    MESSAGE = "could not download file when expect download is true"

    def __init__(self, message: str = MESSAGE):
        super().__init__(message)
        self.message = message
