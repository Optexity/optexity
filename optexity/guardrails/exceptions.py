class GuardrailViolation(PermissionError):
    """Raised when a deterministic guardrail denies an operation."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)
