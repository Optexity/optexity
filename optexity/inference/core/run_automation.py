import asyncio
import logging
import time
import traceback
from copy import deepcopy

from patchright._impl._errors import TimeoutError as PatchrightTimeoutError
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError

from optexity.inference.core.interaction.utils import clean_download
from optexity.inference.core.logging import (
    complete_task_in_server,
    initiate_callback,
    save_downloads_in_server,
    save_latest_memory_state_locally,
    save_output_data_in_server,
    save_trajectory_in_server,
    start_task_in_server,
)
from optexity.inference.core.run_assertion import run_assertion_action
from optexity.inference.core.run_extraction import run_extraction_action
from optexity.inference.core.run_interaction import (
    handle_download_url_as_pdf,
    run_interaction_action,
)
from optexity.inference.core.run_python_script import run_python_script_action
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import DownloadUrlAsPdfAction
from optexity.schema.automation import ActionNode, ForLoopNode, IfElseNode
from optexity.schema.memory import BrowserState, ForLoopStatus, Memory, OutputData
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

# TODO: static check that index for all replacement of input variables are within the bounds of the input variables

# TODO: static check that all for loop expansion for generated variables have some place where generated variables are added to the memory

# TODO: Check that all for loop expansion for generated variables have some place where generated variables are added to the memory

# TODO: give a warning where any variable of type {variable_name[index]} is used but variable_name is not in the memory in generated variables or in input variables

DRIVER_CLOSED_MARKERS = (
    "Connection closed",
    "Target closed",
    "Browser closed",
    "no close frame",
    "has been closed",
    "Target crashed",
)


def is_driver_closed_error(e: Exception) -> bool:
    msg = str(e)
    return any(m in msg for m in DRIVER_CLOSED_MARKERS)


async def run_automation(
    task: Task, unique_child_arn: str, child_process_id: int, max_retries: int = 1
):
    if max_retries <= 0:
        return
    file_handler = logging.FileHandler(str(task.log_file_path))
    file_handler.setLevel(logging.DEBUG)

    current_module = __name__.split(".")[0]  # top-level module/package
    logging.getLogger(current_module).addHandler(file_handler)
    logging.getLogger("browser_use").setLevel(logging.INFO)

    logger.info(f"Task {task.task_id} started running")
    memory = None
    browser = None

    try:
        await start_task_in_server(task)
        memory = Memory(unique_child_arn=unique_child_arn)
        memory.update_system_info()

        def _get_browser():
            return Browser(
                memory=memory,
                headless=False,
                channel=task.automation.browser_channel,
                debug_port=9222 + child_process_id,
                use_proxy=task.use_proxy,
                proxy_session_id=task.proxy_session_id(
                    settings.PROXY_PROVIDER if task.use_proxy else None
                ),
            )

        browser = _get_browser()
        memory.update_system_info()

        automation = task.automation
        memory.automation_state.step_index = -1
        memory.automation_state.try_index = 0

        try:
            await browser.start()
            await browser.go_to_url("about:blank")
            memory.update_system_info()
        except Exception as e:
            logger.error(
                f"Error going to about:blank on start: {e}, stopping browser and restarting"
            )
            await browser.stop(force=True)
            browser = _get_browser()
            await browser.start()
            await browser.go_to_url("about:blank")
            memory.update_system_info()

        if task.use_proxy:

            await browser.go_to_url("https://ipinfo.io/json")
            page = await browser.get_current_page()

            ip_info = await page.evaluate("""
                async () => {
                const res = await fetch("https://ipinfo.io/json");
                return await res.json();
                }
                """)
            if isinstance(ip_info, dict):
                memory.variables.output_data.append(
                    OutputData(unique_identifier="ip_info", json_data=ip_info)
                )
            elif isinstance(ip_info, str):
                memory.variables.output_data.append(
                    OutputData(unique_identifier="ip_info", text=ip_info)
                )
            else:
                try:
                    memory.variables.output_data.append(
                        OutputData(unique_identifier="ip_info", text=str(ip_info))
                    )
                except Exception as e:
                    logger.error(f"Error getting IP info: {e}")

        await browser.go_to_url(task.automation.url)
        memory.update_system_info()

        full_automation = []

        for node in automation.nodes:
            if isinstance(node, ForLoopNode):
                await handle_for_loop_node(node, memory, task, browser, full_automation)
            elif isinstance(node, IfElseNode):
                await handle_if_else_node(node, memory, task, browser, full_automation)
            else:
                full_automation.append(node.model_dump())
                await run_action_node(
                    node,
                    task,
                    memory,
                    browser,
                )

        task.status = "success"
    except AssertionError as e:
        logger.error(f"Assertion error: {e}")
        task.error = str(e)
        task.status = "failed"
    except Exception as e:
        if is_driver_closed_error(e):
            logger.error(f"Driver closed error: {e}, restarting browser")
            if browser is not None:
                await browser.stop(force=True)
        if max_retries > 1:
            logger.info(
                f"Running automations again with {max_retries - 1} retries left"
            )
            return await run_automation(
                task, unique_child_arn, child_process_id, max_retries - 1
            )
        else:
            logger.error(f"Error running automation: {traceback.format_exc()}")
            task.error = str(e)
            task.status = "failed"

    finally:
        if task and memory and browser:
            await run_final_downloads_check(task, memory, browser)
        if memory and browser:
            await run_final_logging(task, memory, browser, child_process_id)
        if browser is not None:
            await browser.stop()

    logger.info(f"Task {task.task_id} completed with status {task.status}")
    file_handler.flush()
    file_handler.close()
    logging.getLogger(current_module).removeHandler(file_handler)


async def run_final_downloads_check(task: Task, memory: Memory, browser: Browser):

    try:
        logger.debug("Running final downloads check")
        max_timeout = 10.0
        start = time.monotonic()
        await asyncio.wait_for(
            browser.all_active_downloads_done.wait(), timeout=max_timeout
        )
        max_timeout = max(0.0, max_timeout - (time.monotonic() - start))

        for temp_download_path, (
            is_downloaded,
            download,
        ) in memory.raw_downloads.items():
            if is_downloaded or download is None:
                continue

            download_path = task.downloads_directory / download.suggested_filename
            await download.save_as(download_path)
            memory.downloads.append(download_path)
            await clean_download(download_path)
            memory.raw_downloads[temp_download_path] = (True, download)

        while max_timeout > 0:
            if (
                len(memory.urls_to_downloads) + len(memory.downloads)
                >= task.automation.expected_downloads
            ):
                break
            interval = min(1, max_timeout)
            await asyncio.sleep(interval)
            max_timeout = max(0.0, max_timeout - interval)

        for url, filename in memory.urls_to_downloads:
            download_path = task.downloads_directory / filename
            await handle_download_url_as_pdf(
                DownloadUrlAsPdfAction(url=url, download_filename=filename),
                task,
                memory,
                browser,
            )

    except Exception as e:
        logger.error(f"Error running final downloads check: {e}")

    logger.warning(
        f"Found {len(memory.downloads)} downloads, expected {task.automation.expected_downloads}"
    )


async def run_final_logging(
    task: Task, memory: Memory, browser: Browser, child_process_id: int
):

    try:
        await complete_task_in_server(task, memory.token_usage, child_process_id)

        try:
            memory.automation_state.step_index += 1
            browser_state_summary = await browser.get_browser_state_summary()
            memory.browser_states.append(
                BrowserState(
                    url=browser_state_summary.url,
                    screenshot=browser_state_summary.screenshot,
                    title=browser_state_summary.title,
                    axtree=browser_state_summary.dom_state.llm_representation(
                        remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
                    ),
                )
            )

            memory.final_screenshot = await browser.get_screenshot(full_page=True)
        except Exception as e:
            logger.error(f"Error getting final screenshot: {e}")

        await save_output_data_in_server(task, memory)
        await save_downloads_in_server(task, memory)
        await save_latest_memory_state_locally(task, memory, None)
        await save_trajectory_in_server(task)
        await initiate_callback(task)

    except Exception as e:
        logger.error(f"Error running final logging: {e}")


async def run_action_node(
    action_node: ActionNode,
    task: Task,
    memory: Memory,
    browser: Browser,
):
    memory.update_system_info()
    await asyncio.sleep(action_node.before_sleep_time)
    await browser.handle_new_tabs(0)

    memory.automation_state.step_index += 1
    memory.automation_state.try_index = 0

    await action_node.replace_variables(task.input_parameters)
    await action_node.replace_variables(task.secure_parameters)
    await action_node.replace_variables(memory.variables.generated_variables)

    # ## TODO: optimize this by taking screenshot and axtree only if needed
    # browser_state_summary = await browser.get_browser_state_summary()

    memory.browser_states.append(
        BrowserState(
            url=await browser.get_current_page_url(),
            screenshot=await browser.get_screenshot(),
            title=await browser.get_current_page_title(),
            axtree=None,
        )
    )

    logger.debug(f"-----Running node {memory.automation_state.step_index}-----")

    try:
        if action_node.interaction_action:
            ## Assuming network calls are only made during interaction actions and not during extraction actions
            await browser.clear_network_calls()

            await run_interaction_action(
                action_node.interaction_action, task, memory, browser, 2
            )
        elif action_node.extraction_action:
            await run_extraction_action(
                action_node.extraction_action, memory, browser, task
            )
        elif action_node.python_script_action:
            await run_python_script_action(
                action_node.python_script_action, memory, browser
            )
        elif action_node.assertion_action:
            await run_assertion_action(
                action_node.assertion_action, memory, browser, task
            )

    except Exception as e:
        logger.error(f"Error running node {memory.automation_state.step_index}: {e}")
        raise e
    finally:
        await save_latest_memory_state_locally(task, memory, action_node)
        if memory.automation_state.step_index % 5 == 0:
            await save_trajectory_in_server(task)

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
    memory.update_system_info()


async def sleep_for_page_to_load(browser: Browser, sleep_time: float):
    await asyncio.sleep(0.1)

    sleep_time = max(0.0, sleep_time - 0.1)

    if float(sleep_time) == 0.0:
        return

    page = await browser.get_current_page()
    if page is None:
        return
    try:
        await page.wait_for_load_state("load", timeout=sleep_time * 1000)
    except (TimeoutError, PatchrightTimeoutError, PlaywrightTimeoutError):
        pass


def evaluate_condition(condition: str, memory: Memory, task: Task) -> bool:
    return eval(
        condition,
        {},
        {**task.input_parameters, **memory.variables.generated_variables},
    )


async def handle_if_else_node(
    if_else_node: IfElseNode,
    memory: Memory,
    task: Task,
    browser: Browser,
    full_automation: list[ActionNode],
):
    memory.update_system_info()
    logger.debug(
        f"Handling if else node {if_else_node.condition} with if nodes {if_else_node.if_nodes} and else nodes {if_else_node.else_nodes}"
    )
    condition_result = evaluate_condition(if_else_node.condition, memory, task)
    if condition_result:
        nodes = if_else_node.if_nodes
    else:
        nodes = if_else_node.else_nodes

    for node in nodes:
        if isinstance(node, ActionNode):
            full_automation.append(node.model_dump())
            await run_action_node(
                node,
                task,
                memory,
                browser,
            )
        elif isinstance(node, IfElseNode):
            await handle_if_else_node(node, memory, task, browser, full_automation)
        elif isinstance(node, ForLoopNode):
            await handle_for_loop_node(node, memory, task, browser, full_automation)

    logger.debug(f"Finished handling if else node {if_else_node.condition}")
    memory.update_system_info()


async def handle_for_loop_node(
    for_loop_node: ForLoopNode,
    memory: Memory,
    task: Task,
    browser: Browser,
    full_automation: list[ActionNode],
):
    memory.update_system_info()
    if for_loop_node.variable_name in task.input_parameters:
        values = task.input_parameters[for_loop_node.variable_name]
    elif for_loop_node.variable_name in memory.variables.generated_variables:
        values = memory.variables.generated_variables[for_loop_node.variable_name]
    else:
        raise ValueError(
            f"Variable name {for_loop_node.variable_name} not found in input variables or generated variables"
        )
    memory.variables.for_loop_status.append([])
    for index in range(len(values)):

        try:
            for node in for_loop_node.nodes:
                new_node = deepcopy(node)
                new_node.replace(
                    f"{{{for_loop_node.variable_name}[index]}}",
                    f"{{{for_loop_node.variable_name}[{index}]}}",
                )
                new_node.replace(
                    f"{{index_of({for_loop_node.variable_name})}}", f"{index}"
                )

                if isinstance(new_node, IfElseNode):
                    await handle_if_else_node(
                        new_node, memory, task, browser, full_automation
                    )

                else:
                    full_automation.append(new_node.model_dump())
                    await run_action_node(
                        new_node,
                        task,
                        memory,
                        browser,
                    )
            memory.variables.for_loop_status[-1].append(
                ForLoopStatus(
                    variable_name=for_loop_node.variable_name,
                    index=index,
                    value=values[index],
                    status="success",
                )
            )
        except Exception as e:
            logger.error(
                f"Error running for loop node {for_loop_node.variable_name}: {e}"
            )
            memory.variables.for_loop_status[-1].append(
                ForLoopStatus(
                    variable_name=for_loop_node.variable_name,
                    index=index,
                    value=values[index],
                    status="error",
                    error=str(e),
                )
            )
            if for_loop_node.on_error_in_loop == "continue":
                continue
            elif for_loop_node.on_error_in_loop == "break":
                for index2 in range(index + 1, len(values)):
                    memory.variables.for_loop_status[-1].append(
                        ForLoopStatus(
                            variable_name=for_loop_node.variable_name,
                            index=index2,
                            value=values[index2],
                            status="skipped",
                        )
                    )

                break
            else:
                raise e

        if index < len(values) - 1:
            for node in for_loop_node.reset_nodes:
                if isinstance(node, IfElseNode):
                    await handle_if_else_node(
                        node, memory, task, browser, full_automation
                    )

                else:
                    full_automation.append(node.model_dump())
                    await run_action_node(
                        node,
                        task,
                        memory,
                        browser,
                    )
    memory.update_system_info()
