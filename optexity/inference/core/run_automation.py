import asyncio
import base64
import json
import logging
from copy import deepcopy
from pathlib import Path

import aiofiles

from optexity.inference.core.run_2fa import run_2fa_action
from optexity.inference.core.run_extraction import run_extraction_action
from optexity.inference.core.run_interaction import run_interaction_action
from optexity.inference.core.run_python_script import run_python_script_action
from optexity.inference.infra.browser import Browser
from optexity.schema.automation import ActionNode, Automation, ForLoopNode, IfElseNode
from optexity.schema.memory import BrowserState, Memory
from optexity.utils.utils import save_screenshot

logger = logging.getLogger(__name__)

# TODO: static check that index for all replacement of input variables are within the bounds of the input variables

# TODO: static check that all for loop expansion for generated variables have some place where generated variables are added to the memory

# TODO: Check that all for loop expansion for generated variables have some place where generated variables are added to the memory

# TODO: give a warning where any variable of type {variable_name[index]} is used but variable_name is not in the memory in generated variables or in input variables


async def run_automation(automation: Automation, memory: Memory, browser: Browser):
    file_handler = logging.FileHandler(str(memory.log_file_path))
    file_handler.setLevel(logging.DEBUG)

    current_module = __name__.split(".")[0]  # top-level module/package
    logging.getLogger(current_module).addHandler(file_handler)
    logging.getLogger("browser_use").setLevel(logging.INFO)

    logger.info(f"Running automation for task {memory.task_id}")

    memory.automation_state.step_index = -1
    memory.automation_state.try_index = 0

    full_automation = []

    for node in automation.nodes:
        if isinstance(node, ForLoopNode):
            action_nodes = expand_for_loop_node(node, memory)
            logger.debug(
                f"Expanded for loop node {node.variable_name} into {len(action_nodes)} nodes"
            )
        elif isinstance(node, IfElseNode):
            action_nodes = handle_if_else_node(node, memory)
            logger.debug(f"nodes for if else node {node.condition} are {action_nodes}")
        else:
            action_nodes = [node]

        for action_node in action_nodes:
            full_automation.append(action_node.model_dump())
            await run_automation_node(action_node, memory, browser)

    memory.automation_state.step_index += 1
    browser_state_summary = await browser.get_browser_state_summary()
    memory.browser_states.append(
        BrowserState(
            url=browser_state_summary.url,
            screenshot=browser_state_summary.screenshot,
            title=browser_state_summary.title,
            axtree=browser_state_summary.dom_state.llm_representation(),
        )
    )

    await save_memory_state(memory, None)

    logging.getLogger(current_module).removeHandler(file_handler)


async def run_automation_node(
    action_node: ActionNode, memory: Memory, browser: Browser
):

    await asyncio.sleep(action_node.before_sleep_time)
    await browser.handle_new_tabs(0)

    memory.automation_state.step_index += 1
    memory.automation_state.try_index = 0

    action_node.replace_variables(memory.variables.input_variables)
    action_node.replace_variables(memory.variables.generated_variables)

    ## TODO: optimize this by taking screenshot and axtree only if needed
    browser_state_summary = await browser.get_browser_state_summary()

    memory.browser_states.append(
        BrowserState(
            url=browser_state_summary.url,
            screenshot=browser_state_summary.screenshot,
            title=browser_state_summary.title,
            axtree=browser_state_summary.dom_state.llm_representation(),
        )
    )
    logger.debug(f"-----Running node {memory.automation_state.step_index}-----")

    try:
        if action_node.interaction_action:
            ## Assuming network calls are only made during interaction actions and not during extraction actions
            await browser.clear_network_calls()
            await browser.attach_network_listeners()

            await run_interaction_action(
                action_node.interaction_action, memory, browser, 2
            )
        elif action_node.extraction_action:
            await run_extraction_action(action_node.extraction_action, memory, browser)
        elif action_node.fetch_2fa_action:
            await run_2fa_action(action_node.fetch_2fa_action, memory, browser)
        elif action_node.python_script_action:
            await run_python_script_action(
                action_node.python_script_action, memory, browser
            )

    except Exception as e:
        logger.error(f"Error running node {memory.automation_state.step_index}: {e}")
        raise e
    finally:

        await save_memory_state(memory, action_node)
    if action_node.expect_new_tab:
        found_new_tab, total_time = await browser.handle_new_tabs(
            action_node.max_new_tab_wait_time
        )
        if not found_new_tab:
            logger.warning(
                f"No new tab found after {action_node.max_new_tab_wait_time} seconds, even though expect_new_tab is True"
            )
        else:
            logger.debug(f"Switched to new tab after {total_time} seconds, as expected")

    else:
        await sleep_for_page_to_load(browser, action_node.end_sleep_time)

    logger.debug(f"-----Finished node {memory.automation_state.step_index}-----")


async def save_memory_state(memory: Memory, node: ActionNode | None):

    browser_state = memory.browser_states[-1]
    automation_state = memory.automation_state
    step_directory = memory.logs_directory / f"step_{str(automation_state.step_index)}"
    step_directory.mkdir(parents=True, exist_ok=True)

    save_screenshot(browser_state.screenshot, step_directory / "screenshot.png")

    state_dict = {
        "title": browser_state.title,
        "url": browser_state.url,
        "step_index": automation_state.step_index,
        "try_index": automation_state.try_index,
        "downloaded_files": [
            downloaded_file.name for downloaded_file in memory.downloads
        ],
        "token_usage": memory.token_usage.model_dump(),
    }

    async with aiofiles.open(step_directory / "state.json", "w") as f:
        await f.write(json.dumps(state_dict, indent=4))

    if browser_state.axtree:
        async with aiofiles.open(step_directory / "axtree.txt", "w") as f:
            await f.write(browser_state.axtree)

    if browser_state.final_prompt:
        async with aiofiles.open(step_directory / "final_prompt.txt", "w") as f:
            await f.write(browser_state.final_prompt)

    if browser_state.llm_response:
        async with aiofiles.open(step_directory / "llm_response.json", "w") as f:
            await f.write(json.dumps(browser_state.llm_response, indent=4))

    if node:
        async with aiofiles.open(step_directory / "action_node.json", "w") as f:
            await f.write(json.dumps(node.model_dump(), indent=4))

    async with aiofiles.open(step_directory / "input_variables.json", "w") as f:
        await f.write(json.dumps(memory.variables.input_variables, indent=4))

    async with aiofiles.open(step_directory / "generated_variables.json", "w") as f:
        await f.write(json.dumps(memory.variables.generated_variables, indent=4))

    async with aiofiles.open(step_directory / "output_data.json", "w") as f:
        await f.write(
            json.dumps(
                [
                    output_data.model_dump(exclude_none=True, exclude={"screenshot"})
                    for output_data in memory.variables.output_data
                ],
                indent=4,
            )
        )

    for output_data in memory.variables.output_data:
        if output_data.screenshot:
            async with aiofiles.open(
                step_directory / f"screenshot_{output_data.screenshot.filename}.png",
                "wb",
            ) as f:
                await f.write(base64.b64decode(output_data.screenshot.base64))


async def save_automation(automation: Automation, save_directory: Path):
    async with aiofiles.open(save_directory / "automation.json", "w") as f:
        await f.write(json.dumps(automation.model_dump(), indent=4))


async def sleep_for_page_to_load(browser: Browser, sleep_time: float):
    page = await browser.get_current_page()
    if page is None:
        return
    try:
        await page.wait_for_load_state("load", timeout=sleep_time * 1000)
    except TimeoutError as e:
        pass


def expand_for_loop_node(
    for_loop_node: ForLoopNode, memory: Memory
) -> list[ActionNode]:

    if for_loop_node.variable_name in memory.variables.input_variables:
        values = memory.variables.input_variables[for_loop_node.variable_name]
    elif for_loop_node.variable_name in memory.variables.generated_variables:
        values = memory.variables.generated_variables[for_loop_node.variable_name]
    else:
        raise ValueError(
            f"Variable name {for_loop_node.variable_name} not found in input variables or generated variables"
        )

    new_nodes = []
    for index in range(len(values)):
        for action_node in for_loop_node.nodes:
            new_node = deepcopy(action_node)
            new_node.replace(
                f"{{{for_loop_node.variable_name}[index]}}",
                f"{{{for_loop_node.variable_name}[{index}]}}",
            )
            new_nodes.append(new_node)

    return new_nodes


def evaluate_condition(condition: str, memory: Memory) -> bool:
    return eval(
        condition,
        {},
        {**memory.variables.input_variables, **memory.variables.generated_variables},
    )


def handle_if_else_node(if_else_node: IfElseNode, memory: Memory) -> list[ActionNode]:
    condition_result = evaluate_condition(if_else_node.condition, memory)
    if condition_result:
        return if_else_node.if_nodes
    else:
        return if_else_node.else_nodes
