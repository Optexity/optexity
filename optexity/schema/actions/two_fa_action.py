from typing import Literal

from pydantic import BaseModel


class EmailTwoFAAction(BaseModel):
    type: Literal["email_two_fa_action"]
    receiver_email_address: str
    sender_email_address: str


class SlackTwoFAAction(BaseModel):
    type: Literal["slack_two_fa_action"]
    slack_workspace_domain: str
    channel_name: str
    sender_name: str


class TwoFAAction(BaseModel):
    action: EmailTwoFAAction | SlackTwoFAAction
    instructions: str | None = None
    output_variable_name: str
    max_wait_time: float = 300.0
    check_interval: float = 10.0
