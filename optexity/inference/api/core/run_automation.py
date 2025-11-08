import asyncio
import json
import logging
from copy import deepcopy

import aiofiles

from optexity.inference.api.core.run_extraction import run_extraction_action
from optexity.inference.api.core.run_interaction import run_interaction_action
from optexity.inference.api.infra.browser import Browser
from optexity.schema.automation import ActionNode, Automation, ForLoopNode
from optexity.schema.memory import BrowserState, Memory

logger = logging.getLogger(__name__)

# TODO: static check that index for all replacement of input variables are within the bounds of the input variables

# TODO: static check that all for loop expansion for generated variables have some place where generated variables are added to the memory

# TODO: Check that all for loop expansion for generated variables have some place where generated variables are added to the memory

# TODO: give a warning where any variable of type {variable_name[index]} is used but variable_name is not in the memory in generated variables or in input variables


async def run_automation(automation: Automation, memory: Memory, browser: Browser):

    # automation = expand_for_loop_based_on_input_variables(automation, memory)

    memory.automation_state.step_index = -1
    memory.automation_state.try_index = 0

    full_automation = []

    for node in automation.nodes:
        if isinstance(node, ForLoopNode):
            action_nodes = expand_for_loop_node(node, memory)
            logger.debug(
                f"Expanded for loop node {node.variable_name} into {len(action_nodes)} nodes"
            )
        else:
            action_nodes = [node]

        for action_node in action_nodes:
            await browser.handle_new_tabs(0)
            memory.automation_state.step_index += 1
            memory.automation_state.try_index = 0

            action_node.replace_variables(memory.variables.input_variables)
            action_node.replace_variables(memory.variables.generated_variables)

            memory.browser_states.append(BrowserState(url=browser.page.url))
            logger.debug(
                f"--------------------------------Running node {memory.automation_state.step_index}--------------------------------"
            )

            full_automation.append(action_node.model_dump())

            try:
                if action_node.interaction_action:
                    await run_interaction_action(
                        action_node.interaction_action, memory, browser
                    )
                # elif action_node.assertion_action:
                #     await browser.run_assertion_action(action_node.assertion_action)
                elif action_node.extraction_action:
                    await run_extraction_action(
                        action_node.extraction_action, memory, browser
                    )

                # elif action_node.python_script_action:
                #     await browser.run_python_script_action(action_node.python_script_action)
            except Exception as e:
                logger.error(
                    f"Error running node {memory.automation_state.step_index}: {e}"
                )
                raise e

            if action_node.expect_new_tab:
                found_new_tab, total_time = await browser.handle_new_tabs(
                    action_node.max_new_tab_wait_time
                )
                if not found_new_tab:
                    logger.warning(
                        f"No new tab found after {action_node.max_new_tab_wait_time} seconds, even though expect_new_tab is True"
                    )
                else:
                    logger.debug(
                        f"Switched to new tab after {total_time} seconds, as expected"
                    )

            else:
                await sleep_for_page_to_load(browser, action_node.end_sleep_time)

            logger.debug(
                f"--------------------------------Finished node {memory.automation_state.step_index}--------------------------------"
            )

            async with aiofiles.open("automation.json", "w") as f:
                await f.write(json.dumps(full_automation, indent=4))

            async with aiofiles.open("memory.json", "w") as f:
                await f.write(json.dumps(memory.model_dump(), indent=4))


async def sleep_for_page_to_load(browser: Browser, sleep_time: float):
    page = await browser.get_current_page()
    if page is None:
        return
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=sleep_time * 1000)
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
