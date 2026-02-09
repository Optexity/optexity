import logging
from pathlib import Path
from typing import Callable

import aiofiles

from optexity.inference.agents.index_prediction.action_prediction_locator_axtree import (
    ActionPredictionLocatorAxtree,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


index_prediction_agent = ActionPredictionLocatorAxtree()


async def get_index_from_prompt(
    memory: Memory, prompt_instructions: str, browser: Browser, task: Task
):
    browser_state_summary = await browser.get_browser_state_summary()
    memory.browser_states[-1] = BrowserState(
        url=browser_state_summary.url,
        screenshot=browser_state_summary.screenshot,
        title=browser_state_summary.title,
        axtree=browser_state_summary.dom_state.llm_representation(
            remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
        ),
    )

    try:
        final_prompt, response, token_usage = index_prediction_agent.predict_action(
            prompt_instructions, memory.browser_states[-1].axtree
        )
        memory.token_usage += token_usage
        memory.browser_states[-1].final_prompt = final_prompt
        memory.browser_states[-1].llm_response = response.model_dump()

        return response.index
    except Exception as e:
        logger.error(f"Error in get_index_from_prompt: {e}")


async def handle_download(
    func: Callable, memory: Memory, browser: Browser, task: Task, download_filename: str
):
    page = await browser.get_current_page()
    if page is None:
        logger.error("No page found for current page")
        return
    download_path: Path = task.downloads_directory / download_filename
    async with page.expect_download() as download_info:
        await func()
        download = await download_info.value

        if download:
            temp_path = await download.path()
            async with memory.download_lock:
                memory.raw_downloads[temp_path] = (True, None)

            # If caller passed a filename with no extension (e.g. UUID only), use
            # Playwright's suggested_filename so the saved file has the correct type.
            if not download_path.suffix and getattr(
                download, "suggested_filename", None
            ):
                suggested = Path(download.suggested_filename)
                if suggested.suffix:
                    download_path = download_path.with_suffix(suggested.suffix)
            await download.save_as(download_path)
            await clean_download(download_path)

            # Detect wrong content: Playwright sometimes captures the wrong response (e.g. HTML
            # page or 0 bytes). Reject so we don't keep a corrupted file. The real PDF response
            # is captured by the response listener (content-disposition attachment + .pdf) and
            # will be fetched in run_final_downloads_check via handle_download_url_as_pdf.
            def _saved_file_is_invalid() -> bool:
                if not download_path.exists():
                    return True
                size = download_path.stat().st_size
                if size == 0:
                    return True
                return False

            if _saved_file_is_invalid():
                try:
                    download_path.unlink(missing_ok=True)
                except OSError:
                    pass
                logger.info(
                    "Discarded invalid download (wrong or empty content); real PDF will be fetched from captured URL"
                )
            else:
                memory.downloads.append(download_path)
        else:
            logger.error("No download found")


async def clean_download(download_path: Path):

    if download_path.suffix == ".csv":
        # Read full file
        async with aiofiles.open(download_path, "r", encoding="utf-8") as f:
            content = await f.read()
        # Remove everything between <script>...</script> (multiline safe)

        if "</script>" in content:
            clean_content = content.split("</script>")[-1]

            # Write cleaned CSV back
            async with aiofiles.open(download_path, "w", encoding="utf-8") as f:
                await f.write(clean_content)
