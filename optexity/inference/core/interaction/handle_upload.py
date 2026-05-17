import logging
import mimetypes
import os
import re
import tempfile
from urllib.parse import unquote, urlparse

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    get_index_from_prompt,
    update_screenshot_with_highlight,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import UploadFileAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_MS = 120_000


def _derive_suffix(
    url: str, content_disposition: str | None, content_type: str | None
) -> str:
    path = urlparse(url).path
    basename = os.path.basename(unquote(path))
    _, ext = os.path.splitext(basename)
    if ext:
        return ext

    if content_disposition:
        match = re.search(
            r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition
        )
        if match:
            _, ext = os.path.splitext(unquote(match.group(1)))
            if ext:
                return ext

    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed

    return ""


async def _download_to_temp_file(url: str, browser: Browser) -> str:
    logger.debug(f"Downloading upload file from {url}")
    try:
        resp = await browser.context.request.get(url, timeout=_DOWNLOAD_TIMEOUT_MS)
    except Exception as e:
        raise RuntimeError(f"Failed to download upload file from {url}: {e}") from e

    if not resp.ok:
        raise RuntimeError(
            f"Failed to download upload file from {url}: HTTP {resp.status}"
        )

    headers = resp.headers
    suffix = _derive_suffix(
        url, headers.get("content-disposition"), headers.get("content-type")
    )

    body = await resp.body()
    with tempfile.NamedTemporaryFile(
        prefix="optexity_upload_", suffix=suffix, delete=False
    ) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    logger.debug(f"Downloaded upload file to {tmp_path} ({len(body)} bytes)")
    return tmp_path


async def handle_upload_file(
    upload_file_action: UploadFileAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):
    tmp_path: str | None = None
    if upload_file_action.file_url:
        tmp_path = await _download_to_temp_file(upload_file_action.file_url, browser)
        upload_file_action.file_path = tmp_path

    try:
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
            await upload_file_index(upload_file_action, browser, memory, task)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp upload file {tmp_path}: {e}")


async def upload_file_index(
    upload_file_action: UploadFileAction, browser: Browser, memory: Memory, task: Task
):

    try:
        index = await get_index_from_prompt(
            memory, upload_file_action.prompt_instructions, browser, task
        )
        if index is None:
            return
        try:
            await update_screenshot_with_highlight(browser, memory, index)
        except Exception as e:
            logger.error(
                f"Error in updating screenshot with highlight in upload_file_index: {e}"
            )

        action_model = browser.backend_agent.ActionModel(
            **{"upload_file": {"index": index, "path": upload_file_action.file_path}}
        )
        await browser.backend_agent.multi_act([action_model])
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in upload_file_index: {e}")
        return
