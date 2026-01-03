from typing import Literal

from pydantic import BaseModel


class EmailTwoFactorAuthAction(BaseModel):
    type: Literal["email_two_factor_auth"] = "email_two_factor_auth"
    receiver_email_address: str
    sender_email_address: str


class SlackTwoFactorAuthAction(BaseModel):
    type: Literal["slack_two_factor_auth"] = "slack_two_factor_auth"
    slack_workspace_domain: str
    channel_name: str
    sender_name: str


class TwoFactorAuthAction(BaseModel):
    action: EmailTwoFactorAuthAction | SlackTwoFactorAuthAction
    output_variable_name: str
    max_wait_time: float = 300.0
