import logging

from optexity.inference.core.two_factor_auth.gmail import get_code_from_gmail
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.two_factor_auth_action import (
    Fetch2faAction,
    Fetch2faApiCallAction,
    FetchEmail2FaAction,
    FetchTotp2FaAction,
)
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)


async def run_2fa_action(
    fetch_2fa_action: Fetch2faAction, memory: Memory, browser: Browser
):
    logger.debug(
        f"---------Running 2fa action {fetch_2fa_action.model_dump_json()}---------"
    )

    if fetch_2fa_action.fetch_email_2fa_action:
        code_2fa = await handle_fetch_email_2fa(
            fetch_2fa_action.fetch_email_2fa_action, memory, browser
        )
    elif fetch_2fa_action.fetch_totp_2fa_action:
        code_2fa = await handle_fetch_totp_2fa(
            fetch_2fa_action.fetch_totp_2fa_action, memory, browser
        )
    elif fetch_2fa_action.fetch_2fa_api_call_action:
        code_2fa = await handle_fetch_2fa_api_call(
            fetch_2fa_action.fetch_2fa_api_call_action, memory, browser
        )

    memory.automation_state.start_2fa_time = None
    if code_2fa is None:
        raise ValueError("No 2FA code found")

    memory.variables.generated_variables[fetch_2fa_action.output_variable_name] = (
        code_2fa
    )

    return code_2fa


async def handle_fetch_email_2fa(
    fetch_email_2fa_action: FetchEmail2FaAction, memory: Memory, browser: Browser
):
    if fetch_email_2fa_action.service == "gmail":
        code_2fa = get_code_from_gmail(
            fetch_email_2fa_action.email_address,
            fetch_email_2fa_action.subject,
            memory.automation_state.start_2fa_time,
        )
    else:
        raise ValueError(f"Service {fetch_email_2fa_action.service} not supported")

    return code_2fa


async def handle_fetch_totp_2fa(
    fetch_totp_2fa_action: FetchTotp2FaAction, memory: Memory, browser: Browser
):
    pass


async def handle_fetch_2fa_api_call(
    fetch_2fa_api_call_action: Fetch2faApiCallAction, memory: Memory, browser: Browser
):
    raise NotImplementedError("Fetch 2FA API call is not implemented")
