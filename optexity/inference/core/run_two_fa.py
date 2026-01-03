import logging
from datetime import timedelta
from urllib.parse import urljoin

import httpx

from optexity.schema.actions.two_fa_action import (
    EmailTwoFAAction,
    SlackTwoFAAction,
    TwoFAAction,
)
from optexity.schema.inference import (
    FetchEmailMessagesRequest,
    FetchMessagesResponse,
    FetchSlackMessagesRequest,
)
from optexity.schema.memory import Memory
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)


async def run_two_fa_action(two_fa_action: TwoFAAction, memory: Memory):
    logger.debug(
        f"---------Running 2fa action {two_fa_action.model_dump_json()}---------"
    )

    if isinstance(two_fa_action.action, EmailTwoFAAction):
        code = await handle_email_two_fa(
            two_fa_action.action, memory, two_fa_action.max_wait_time
        )
    elif isinstance(two_fa_action.action, SlackTwoFAAction):
        code = await handle_slack_two_fa(
            two_fa_action.action, memory, two_fa_action.max_wait_time
        )

    memory.automation_state.start_2fa_time = None
    if code is None:
        raise ValueError("No 2FA code found")

    memory.variables.generated_variables[two_fa_action.output_variable_name] = code

    return code


async def handle_email_two_fa(
    email_two_fa_action: EmailTwoFAAction,
    memory: Memory,
    max_wait_time: float,
):

    async with httpx.AsyncClient() as client:
        url = urljoin(settings.SERVER_URL, settings.FETCH_EMAIL_MESSAGES_ENDPOINT)

        body = FetchEmailMessagesRequest(
            receiver_email_address=email_two_fa_action.receiver_email_address,
            sender_email_address=email_two_fa_action.sender_email_address,
            start_2fa_time=memory.automation_state.start_2fa_time,
            end_2fa_time=memory.automation_state.start_2fa_time
            + timedelta(seconds=max_wait_time),
        )
        response = await client.post(url, json=body.model_dump())
        response.raise_for_status()
        response_data = FetchMessagesResponse.model_validate_json(response.json())

        return response_data.messages


async def handle_slack_two_fa(
    slack_two_fa_action: SlackTwoFAAction,
    memory: Memory,
    max_wait_time: float,
):
    async with httpx.AsyncClient() as client:
        url = urljoin(settings.SERVER_URL, settings.FETCH_SLACK_MESSAGES_ENDPOINT)

        body = FetchSlackMessagesRequest(
            slack_workspace_domain=slack_two_fa_action.slack_workspace_domain,
            channel_name=slack_two_fa_action.channel_name,
            sender_name=slack_two_fa_action.sender_name,
            start_2fa_time=memory.automation_state.start_2fa_time,
            end_2fa_time=memory.automation_state.start_2fa_time
            + timedelta(seconds=max_wait_time),
        )
        response = await client.post(url, json=body.model_dump())
        response.raise_for_status()
        response_data = FetchMessagesResponse.model_validate_json(response.json())

        return response_data.messages
