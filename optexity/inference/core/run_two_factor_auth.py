import logging
from datetime import timedelta
from urllib.parse import urljoin

import httpx

from optexity.inference.infra.browser import Browser
from optexity.schema.actions.two_factor_auth_action import (
    EmailTwoFactorAuthAction,
    SlackTwoFactorAuthAction,
    TwoFactorAuthAction,
)
from optexity.schema.inference import (
    FetchOTPFromEmailRequest,
    FetchOTPFromSlackRequest,
    FetchOTPResponse,
)
from optexity.schema.memory import Memory
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)


async def run_two_factor_auth_action(
    two_factor_auth_action: TwoFactorAuthAction, memory: Memory
):
    logger.debug(
        f"---------Running 2fa action {two_factor_auth_action.model_dump_json()}---------"
    )

    if isinstance(two_factor_auth_action.action, EmailTwoFactorAuthAction):
        code = await handle_email_two_factor_auth(
            two_factor_auth_action.action, memory, two_factor_auth_action.max_wait_time
        )
    elif isinstance(two_factor_auth_action.action, SlackTwoFactorAuthAction):
        code = await handle_slack_two_factor_auth(
            two_factor_auth_action.action, memory, two_factor_auth_action.max_wait_time
        )

    memory.automation_state.start_2fa_time = None
    if code is None:
        raise ValueError("No 2FA code found")

    memory.variables.generated_variables[
        two_factor_auth_action.output_variable_name
    ] = code

    return code


async def handle_email_two_factor_auth(
    email_two_factor_auth_action: EmailTwoFactorAuthAction,
    memory: Memory,
    max_wait_time: float,
):

    async with httpx.AsyncClient() as client:
        url = urljoin(settings.SERVER_URL, settings.FETCH_OTP_FROM_EMAIL_ENDPOINT)

        body = FetchOTPFromEmailRequest(
            receiver_email_address=email_two_factor_auth_action.receiver_email_address,
            sender_email_address=email_two_factor_auth_action.sender_email_address,
            start_2fa_time=memory.automation_state.start_2fa_time,
            end_2fa_time=memory.automation_state.start_2fa_time
            + timedelta(seconds=max_wait_time),
        )
        response = await client.post(url, json=body.model_dump())
        response.raise_for_status()
        response_data = FetchOTPResponse.model_validate_json(response.json())

        return response_data.otp


async def handle_slack_two_factor_auth(
    slack_two_factor_auth_action: SlackTwoFactorAuthAction,
    memory: Memory,
    max_wait_time: float,
):
    async with httpx.AsyncClient() as client:
        url = urljoin(settings.SERVER_URL, settings.FETCH_OTP_FROM_SLACK_ENDPOINT)

        body = FetchOTPFromSlackRequest(
            slack_workspace_domain=slack_two_factor_auth_action.slack_workspace_domain,
            channel_name=slack_two_factor_auth_action.channel_name,
            sender_name=slack_two_factor_auth_action.sender_name,
            start_2fa_time=memory.automation_state.start_2fa_time,
            end_2fa_time=memory.automation_state.start_2fa_time
            + timedelta(seconds=max_wait_time),
        )
        response = await client.post(url, json=body.model_dump())
        response.raise_for_status()
        response_data = FetchOTPResponse.model_validate_json(response.json())

        return response_data.otp
