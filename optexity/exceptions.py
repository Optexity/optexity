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
