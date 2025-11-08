import asyncio
import logging

from optexity.inference.api.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)


def get_index_from_prompt(prompt: str, axtree: str) -> int | None:
    raise NotImplementedError("Not implemented")


async def run_interaction_action(
    interaction_action: InteractionAction, memory: Memory, browser: Browser
):
    logger.debug(
        f"---------Running interaction action {interaction_action.model_dump_json()}---------"
    )

    if interaction_action.click_element:
        await handle_click_element(interaction_action.click_element, memory, browser)
    elif interaction_action.input_text:
        await handle_input_text(interaction_action.input_text, memory, browser)
    elif interaction_action.select_option:
        await handle_select_option(interaction_action.select_option, memory, browser)
    elif interaction_action.go_back:
        await handle_go_back(interaction_action.go_back, memory, browser)


async def handle_click_element(
    click_element_action: ClickElementAction, memory: Memory, browser: Browser
):
    max_tries = 10
    max_timeout_per_try = 1
    if click_element_action.command:
        for try_index in range(max_tries):
            try:
                locator = await browser.get_locator_from_command(
                    click_element_action.command
                )
                try:
                    if click_element_action.double_click:
                        await locator.dblclick(timeout=max_timeout_per_try * 1000)
                    else:
                        await locator.click(timeout=max_timeout_per_try * 1000)
                    logger.debug(
                        f"Click element successful after {try_index + 1} tries"
                    )
                    return
                except TimeoutError as e:
                    asyncio.sleep(max_timeout_per_try)

                # if download_filename:
                #     await self.expect_download(download_filename)

            except Exception as e:
                last_error = e

        logger.debug(f"Error in click_element_locator: {last_error}")
        logger.debug("Falling back to index locator")

    if (
        click_element_action.prompt_instructions
        and not click_element_action.skip_prompt
    ):
        memory.automation_state.try_index += 1
        try:
            axtree = memory.browser_states[-1].axtree
            index = get_index_from_prompt(
                click_element_action.prompt_instructions, axtree
            )
            await browser.click_index(index)
            return
        except Exception as e:
            if click_element_action.assert_locator_presence:
                logger.debug(
                    f"Raising error as locator not present and assert_locator_presence is True: {click_element_action.command}"
                )
                raise e
            logger.error(f"Error in get_index_from_prompt: {e}")
            logger.debug("Falling back to index locator")


async def handle_input_text(
    input_text_action: InputTextAction, memory: Memory, browser: Browser
):
    if input_text_action.command:
        try:
            locator = await browser.get_locator_from_command(input_text_action.command)

            if input_text_action.fill_or_type == "fill":
                await locator.fill(input_text_action.input_text)
            else:
                await locator.type(input_text_action.input_text, delay=100)

            return
        except Exception as e:
            logger.debug(f"Error in click_element_locator: {e}")
            logger.debug("Falling back to index locator")

    if input_text_action.prompt_instructions and not input_text_action.skip_prompt:
        memory.automation_state.try_index += 1
        try:
            axtree = memory.browser_states[-1].axtree
            index = get_index_from_prompt(input_text_action.prompt_instructions, axtree)
            await browser.input_text_index(index, input_text_action.input_text)
            return
        except Exception as e:
            if input_text_action.assert_locator_presence:
                logger.debug(
                    f"Raising error as locator not present and assert_locator_presence is True: {input_text_action.command}, {input_text_action.input_text}"
                )
                raise e
            logger.error(f"Error in get_index_from_prompt: {e}")
            logger.debug("Falling back to index locator")


async def handle_select_option(
    select_option_action: SelectOptionAction, memory: Memory, browser: Browser
):
    if select_option_action.command:
        try:
            print(f"Selecting option: {select_option_action.select_values}")
            locator = await browser.get_locator_from_command(
                select_option_action.command
            )
            await locator.select_option(select_option_action.select_values)
            return
        except Exception as e:
            if select_option_action.assert_locator_presence:
                logger.debug(
                    f"Raising error as locator not present and assert_locator_presence is True: {select_option_action.command}, {select_option_action.select_values}"
                )
                raise e
            logger.debug(f"Error in select_option_locator: {e}")
            logger.debug("Falling back to index locator")


async def handle_go_back(
    go_back_action: GoBackAction, memory: Memory, browser: Browser
):
    page = await browser.get_current_page()
    if page is None:
        return
    await page.go_back()
