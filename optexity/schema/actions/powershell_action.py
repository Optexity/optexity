from pydantic import BaseModel, model_validator


class PowerShellAction(BaseModel):
    """Run a list of PowerShell commands on the current RDP Windows machine.

    Opens PowerShell via Win+R, executes all commands sequentially,
    and closes the session (sends 'exit' automatically).
    """

    commands: list[str]

    @model_validator(mode="after")
    def validate_commands(self):
        if not self.commands:
            raise ValueError("At least one command must be provided")
        return self

    def replace(self, pattern: str, replacement: str):
        self.commands = [cmd.replace(pattern, replacement) for cmd in self.commands]
        return self
