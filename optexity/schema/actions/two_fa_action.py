from typing import Literal

from pydantic import BaseModel


class EmailTwoFAAction(BaseModel):
    type: Literal["email_two_fa_action"]
    receiver_email_address: str
    sender_email_address: str

    def replace(self, pattern: str, replacement: str):
        if self.receiver_email_address:
            self.receiver_email_address = self.receiver_email_address.replace(
                pattern, replacement
            )
        if self.sender_email_address:
            self.sender_email_address = self.sender_email_address.replace(
                pattern, replacement
            )


class SlackTwoFAAction(BaseModel):
    type: Literal["slack_two_fa_action"]
    slack_workspace_domain: str
    channel_name: str
    sender_name: str

    def replace(self, pattern: str, replacement: str):
        if self.slack_workspace_domain:
            self.slack_workspace_domain = self.slack_workspace_domain.replace(
                pattern, replacement
            )
        if self.channel_name:
            self.channel_name = self.channel_name.replace(pattern, replacement)
        if self.sender_name:
            self.sender_name = self.sender_name.replace(pattern, replacement)


class TwoFAAction(BaseModel):
    action: EmailTwoFAAction | SlackTwoFAAction
    instructions: str | None = None
    output_variable_name: str
    max_wait_time: float = 300.0
    check_interval: float = 30.0

    def replace(self, pattern: str, replacement: str):
        if self.instructions:
            self.instructions = self.instructions.replace(pattern, replacement)
        if self.action:
            self.action.replace(pattern, replacement)
        return self
