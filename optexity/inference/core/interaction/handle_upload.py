import logging

from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import get_index_from_prompt
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import UploadFileAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_upload_file(
    upload_file_action: UploadFileAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):
    if upload_file_action.command and not upload_file_action.skip_command:
        last_error = await command_based_action_with_retry(
            upload_file_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )
        if last_error is None:
            return

    if not upload_file_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {upload_file_action.__class__.__name__}"
        )
        await upload_file_index(upload_file_action, browser, memory)


async def upload_file_index(
    upload_file_action: UploadFileAction, browser: Browser, memory: Memory
):

    try:
        index = await get_index_from_prompt(
            memory, upload_file_action.prompt_instructions, browser
        )
        if index is None:
            return

        action_model = browser.backend_agent.ActionModel(
            **{"upload_file": {"index": index, "path": upload_file_action.file_path}}
        )
        await browser.backend_agent.multi_act([action_model])
    except Exception as e:
        logger.error(f"Error in upload_file_index: {e}")
        return
