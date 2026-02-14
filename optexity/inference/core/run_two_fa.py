import asyncio
import logging
from datetime import timedelta
from urllib.parse import urljoin

import httpx

from optexity.inference.agents.two_fa_extraction.two_fa_extraction import (
    TwoFAExtraction,
)
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
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

two_fa_extraction_agent = TwoFAExtraction()


async def run_two_fa_action(two_fa_action: TwoFAAction, memory: Memory, task: Task):
    logger.debug(
        f"---------Running 2fa action {two_fa_action.model_dump_json()}---------"
    )

    elapsed = 0
    messages = None
    code = None

    while elapsed < two_fa_action.max_wait_time:
        messages = await fetch_messages(
            two_fa_action.action, memory, two_fa_action.max_wait_time, task
        )
        if messages and len(messages) > 0:
            final_prompt, response, token_usage = two_fa_extraction_agent.extract_code(
                two_fa_action.instructions, messages
            )
            memory.token_usage += token_usage
            code = None
            if response.code is not None:
                if isinstance(response.code, str):
                    code = response.code
                elif isinstance(response.code, list):
                    if len(response.code) > 0:
                        raise ValueError(f"Multiple 2FA codes found, {response.code}")
                    else:
                        code = response.code[0]

            if code is not None:
                logger.debug(
                    f"2FA code {code} found after {elapsed} seconds from {messages}"
                )
                break
            logger.debug(
                f"No 2FA code found in messages, {messages}, waiting for {two_fa_action.check_interval} seconds"
            )
        else:
            logger.debug(
                f"No messages found for 2FA code after {elapsed} seconds, waiting for {two_fa_action.check_interval} seconds"
            )

        await asyncio.sleep(two_fa_action.check_interval)
        elapsed += two_fa_action.check_interval

    memory.automation_state.start_2fa_time = None
    if code is None:
        raise ValueError("2FA code not found")

    memory.variables.generated_variables[two_fa_action.output_variable_name] = [code]

    return code


async def fetch_messages(
    action: EmailTwoFAAction | SlackTwoFAAction,
    memory: Memory,
    max_wait_time: float,
    task: Task,
):

    start_2fa_time = memory.automation_state.start_2fa_time
    end_2fa_time = memory.automation_state.start_2fa_time + timedelta(
        seconds=max_wait_time
    )

    headers = {"x-api-key": task.api_key}

    if isinstance(action, EmailTwoFAAction):
        url = urljoin(settings.SERVER_URL, settings.FETCH_EMAIL_MESSAGES_ENDPOINT)
        body = FetchEmailMessagesRequest(
            receiver_email_address=action.receiver_email_address,
            sender_email_address=action.sender_email_address,
            start_2fa_time=start_2fa_time,
            end_2fa_time=end_2fa_time,
        )
    elif isinstance(action, SlackTwoFAAction):
        url = urljoin(settings.SERVER_URL, settings.FETCH_SLACK_MESSAGES_ENDPOINT)
        body = FetchSlackMessagesRequest(
            slack_workspace_domain=action.slack_workspace_domain,
            channel_name=action.channel_name,
            sender_name=action.sender_name,
            start_2fa_time=start_2fa_time,
            end_2fa_time=end_2fa_time,
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:

            response = await client.post(
                url, json=body.model_dump(mode="json"), headers=headers
            )
            response.raise_for_status()
            response_data = FetchMessagesResponse.model_validate(response.json())

            return response_data.messages
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        return []
