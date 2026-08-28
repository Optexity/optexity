import argparse
import asyncio
import importlib
import inspect
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

import httpx
import psutil
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn import run

from optexity.inference.core.logging import (
    complete_task_in_server,
    delete_local_data,
    initiate_callback,
    save_downloads_in_server,
    save_output_data_in_server,
    save_trajectory_in_server,
    start_task_in_server,
)
from optexity.inference.infra.actual_browser import ActualBrowser
from optexity.inference.infra.browser_health import consume_browser_restart_request
from optexity.schema.automation import Automation
from optexity.schema.enums import ExitCodes
from optexity.schema.inference import InferenceRequest
from optexity.schema.memory import Memory, SystemInfo
from optexity.schema.task import Task
from optexity.utils.http import request_with_backoff
from optexity.utils.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChildProcessIdRequest(BaseModel):
    new_child_process_id: str
    new_unique_child_arn: str


class HumanInLoopCompletedBody(BaseModel):
    task_id: str


child_process_id = -1
unique_child_arn: str = str(uuid.uuid4())
task_running = False
last_task_start_time: datetime | None = None
current_task_timeout_minutes: int | None = None
task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
# Monotonic tie-breaker so equal-priority entries stay FIFO and heap ordering
# never compares Task objects. See Task.priority_order_key.
_task_seq = 0
tasks_to_kill: set[str] = set()


def _enqueue_task(task: Task) -> None:
    """Put a task on the local priority queue: lower priority runs first, None
    last, ties FIFO. The queue is unbounded so put_nowait never blocks."""
    global _task_seq
    _task_seq += 1
    task_queue.put_nowait((*task.priority_order_key(), _task_seq, task))


# task_id -> worker subprocess, so /kill_task can signal an in-flight worker.
running_task_processes: dict[str, asyncio.subprocess.Process] = {}
_global_actual_browser: ActualBrowser | None = None

# HITL: task_ids whose HITL step has been completed by the human.
# Written by POST /human_in_loop_completed; read + cleared by GET /hitl_status.
hitl_completed_tasks: set[str] = set()

# Port this FastAPI server is listening on; set in get_app_with_endpoints so
# it can be forwarded to worker subprocesses via CHILD_FASTAPI_PORT env var.
_child_fastapi_port: int = -1


def log_system_info(comment: str):
    logger.info("=" * 100 + "\n")
    logger.info(comment)
    system_info = SystemInfo()
    logger.info(
        json.dumps(
            {
                "container_memory_total": round(system_info.total_system_memory, 2),
                "container_memory_used": round(system_info.total_system_memory_used, 2),
                "percent_container_memory_used": round(
                    system_info.total_system_memory_used
                    / system_info.total_system_memory,
                    2,
                ),
            }
        )
    )
    vm = psutil.virtual_memory()
    logger.info(
        json.dumps(
            {
                "host_memory_total": round(vm.total / (1024**2), 2),
                "host_memory_used": round(vm.used / (1024**2), 2),
                "percent_host_memory_used": round(vm.used / vm.total, 2),
            }
        )
    )
    logger.info("=" * 100 + "\n")


async def restart_global_actual_browser(reason: str) -> None:
    global _global_actual_browser
    logger.warning("Restarting actual browser: %s", reason)
    if _global_actual_browser is not None:
        try:
            await _global_actual_browser.stop(graceful=True)
        except Exception as e:
            logger.warning("Error stopping browser during restart: %s", e)
        _global_actual_browser = None


async def setup_browser(task: Task, unique_child_arn: str, child_process_id: int):
    assert task.automation is not None, f"Task {task.task_id} has no automation"
    global _global_actual_browser
    system_info = SystemInfo()
    memory_exceeded = (
        system_info.total_system_memory_used / system_info.total_system_memory > 0.6
    )

    # Drain any pending restart flag first so it can't leak into a subsequent task
    # if the global browser was already nulled out (e.g. by the outer-finally restart
    # after WORKER_CRASHED / timeout, or by the retry path in _run_attempt).
    restart_reason = consume_browser_restart_request(child_process_id)
    if restart_reason and _global_actual_browser is None:
        logger.info(
            "Discarding stale browser restart request (browser already absent): %s",
            restart_reason[:500],
        )
        restart_reason = None

    # The health checks below navigate to about:blank by default, which would
    # discard the page this task is meant to resume on. Keep the page only when
    # the automation opted in, so every other run keeps the existing probe.
    preserve_page = bool(
        task.is_dedicated and task.automation.reuse_page_if_already_on_url
    )

    if _global_actual_browser is not None:
        restart_browser = False
        if restart_reason:
            logger.info(
                "Worker requested browser restart before task: %s", restart_reason[:500]
            )
            restart_browser = True

        if not await _global_actual_browser.check_browser_alive(
            preserve_page=preserve_page
        ):
            logger.info("CDP is not alive, restarting browser")
            restart_browser = True

        if task.is_dedicated and not restart_browser:
            if not await _global_actual_browser.check_browser_session_healthy(
                preserve_page=preserve_page
            ):
                logger.info("Dedicated browser session unhealthy, restarting browser")
                restart_browser = True

        if memory_exceeded:
            logger.info("Memory exceeded, restarting browser")
            restart_browser = True

        if not task.is_dedicated:
            logger.info("Previous browser was not dedicated, restarting browser")
            restart_browser = True

        if restart_browser:
            await restart_global_actual_browser(
                restart_reason or "setup_browser health check"
            )

    if _global_actual_browser is None:
        logger.info("Starting new actual browser")
        _global_actual_browser = ActualBrowser(
            channel=task.automation.browser_channel,
            unique_child_arn=unique_child_arn,
            port=9222 + child_process_id,
            headless=False,
            is_dedicated=task.is_dedicated,
            use_proxy=task.use_proxy,
            proxy_session_id=task.proxy_session_id(
                settings.PROXY_PROVIDER if task.use_proxy else None
            ),
            os_emulation=task.automation.os_emulation,
            allow_cookies=task.automation.allow_cookies,
        )
        try:
            await _global_actual_browser.start()
        except Exception:
            logger.exception(
                "Failed to start actual browser; resetting browser instance"
            )
            _global_actual_browser = None
            raise


async def run_automation_in_process(
    task: Task, unique_child_arn: str, child_process_id: int
):

    global _global_actual_browser

    file_handler = logging.FileHandler(str(task.log_file_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s"
        )
    )

    current_module = __name__.split(".")[0]  # top-level module/package
    logging.getLogger(current_module).addHandler(file_handler)
    logger.info(
        f"---------- Starting to run automation for task {task.task_id} ----------\n"
    )
    assert task.automation is not None, f"Task {task.task_id} has no automation"
    worker_path = pathlib.Path(__file__).parent / "worker.py"
    total_attempts = max(1, int(task.automation.max_retries) + 1)
    returncode: int | None = None

    async def _run_attempt(attempt_index: int) -> int | None:
        global _global_actual_browser
        nonlocal returncode

        attempts_left = total_attempts - attempt_index
        task.retry_count = attempt_index

        log_system_info("Memory info before starting browser")
        await setup_browser(task, unique_child_arn, child_process_id)
        log_system_info("Memory info after starting browser")

        if _global_actual_browser is None:
            raise ValueError("Browser is not setup")
        _cdp_url = _global_actual_browser.cdp_url
        if _cdp_url is None:
            raise ValueError("CDP URL is not setup")

        logger.info(
            f"Starting worker attempt {attempt_index + 1}/{total_attempts} (attempts_left={attempts_left})"
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            worker_path,
            task.model_dump_json(),
            unique_child_arn,
            str(child_process_id),
            str(_cdp_url),
            str(attempts_left),
            preexec_fn=os.setsid,
            env={
                **os.environ,
                "CHILD_FASTAPI_PORT": str(_child_fastapi_port),
                "CHILD_PROCESS_ID": str(child_process_id),
            },
        )
        running_task_processes[task.task_id] = proc

        try:
            try:
                logger.debug("Waiting for automation to finish")
                returncode = await asyncio.wait_for(
                    proc.wait(), timeout=task.max_timeout_in_minutes * 60
                )
                logger.info(f"Worker finished with return code {returncode}")
            except asyncio.TimeoutError:
                logger.info(
                    f"Automation timed out after {task.max_timeout_in_minutes} minutes in process"
                )
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception as exc:
                    logger.warning(
                        f"Failed to SIGKILL worker process group for task "
                        f"{task.task_id} after timeout: {exc}"
                    )
                task.status = "killed"
                task.error = f"Automation timed out after {task.max_timeout_in_minutes} minutes in process"
                if attempts_left <= 1:
                    task.completed_at = datetime.now(timezone.utc)
                    await complete_task_in_server(
                        task, None, child_process_id, unique_child_arn
                    )
                    await initiate_callback(task)
                returncode = -1
        finally:
            running_task_processes.pop(task.task_id, None)

        # If the task was cancelled (via /kill_task) while the worker was running,
        # the subprocess has been killed from under us. Report cancellation and
        # skip the retry loop.
        if task.task_id in tasks_to_kill:
            tasks_to_kill.discard(task.task_id)
            task.status = "cancelled"
            task.error = "Task cancelled by user"
            task.completed_at = datetime.now(timezone.utc)
            await complete_task_in_server(
                task, None, child_process_id, unique_child_arn
            )
            return returncode

        if returncode == ExitCodes.SUCCESS.value:
            return returncode

        if attempts_left <= 1:
            return returncode

        # Backoff before retrying.
        sleep_time = 10 * 2**attempt_index
        logger.info(
            f"Retrying automation in process after {sleep_time} seconds (attempts_left={attempts_left - 1})"
        )
        await asyncio.sleep(sleep_time)

        # Force a browser restart before the next attempt (helps with crashed/poisoned sessions).
        if _global_actual_browser is not None:
            try:
                await _global_actual_browser.stop(graceful=True)
            except Exception:
                pass
            _global_actual_browser = None

        return await _run_attempt(attempt_index + 1)

    returncode: int | None = None
    try:
        returncode = await _run_attempt(0)
    finally:
        logger.info(
            f"---------- Automation for task {task.task_id} finished ----------\n"
        )
        log_system_info("Memory info after automation finished in process")

        if (
            task.is_dedicated
            and returncode in (ExitCodes.WORKER_CRASHED.value, -1)
            and _global_actual_browser is not None
        ):
            reason = "timeout" if returncode == -1 else "worker crash"
            await restart_global_actual_browser(
                f"dedicated browser restart after {reason} on task {task.task_id}"
            )

        if _global_actual_browser is not None and not task.is_dedicated:
            logger.debug("Stopping actual browser as not dedicated")
            try:
                await _global_actual_browser.stop(graceful=True)
                _global_actual_browser = None
            except Exception as e:
                logger.error(f"Error stopping actual browser: {e}")

        log_system_info("Memory info after stopping actual browser")

        file_handler.flush()
        file_handler.close()
        logging.getLogger(current_module).removeHandler(file_handler)

        await save_trajectory_in_server(task)
        await delete_local_data(task)


MARKETPLACE_ENDPOINT_PREFIX = "marketplace_"
MARKETPLACE_FUNCTION_MARKER = "fn_"
MARKETPLACE_AUTOMATION_MARKER = "automation_"


def parse_marketplace_endpoint_name(endpoint_name: str) -> tuple[str, str, str] | None:
    """Split a marketplace endpoint name into ``(kind, portal, target)``.

    The namespace states its kind rather than implying it by omission:

        marketplace_<portal>_fn_<function>          -> ("fn", portal, function)
        marketplace_<portal>_automation_<workflow>  -> ("automation", portal, workflow)

    Returns None for anything malformed, so a bad name is reported as such
    instead of being dispatched half-parsed. Mirrored in opcloud's
    `shared/marketplace.py` (this package cannot import it); keep both in step.
    """
    if not endpoint_name.startswith(MARKETPLACE_ENDPOINT_PREFIX):
        return None
    rest = endpoint_name[len(MARKETPLACE_ENDPOINT_PREFIX) :]
    # The portal is always the first segment; the kind marker follows it.
    portal, _, remainder = rest.partition("_")
    if not portal or not remainder:
        return None
    if remainder.startswith(MARKETPLACE_FUNCTION_MARKER):
        target = remainder[len(MARKETPLACE_FUNCTION_MARKER) :]
        return ("fn", portal, target) if target else None
    if remainder.startswith(MARKETPLACE_AUTOMATION_MARKER):
        target = remainder[len(MARKETPLACE_AUTOMATION_MARKER) :]
        return ("automation", portal, target) if target else None
    return None


async def run_marketplace_function(
    task: Task, unique_child_arn: str, child_process_id: int
) -> None:
    """Call a single `optexity_private.portals.<portal>.methods` function
    directly, with no browser/node graph — for endpoint_name
    "marketplace_<portal>_fn_<function_name>". Cookie-requiring functions get
    `processed_cookie_data` fetched from the DB here, never from the caller.
    The dispatched function must be `async def` and return `(success, message)`.
    """
    memory: Memory | None = None
    try:
        parsed = parse_marketplace_endpoint_name(task.endpoint_name)
        if parsed is None or parsed[0] != "fn":
            raise ValueError(
                f"'{task.endpoint_name}' is not a marketplace function endpoint "
                "(expected marketplace_<portal>_fn_<function_name>)"
            )
        _, portal, function_name = parsed
        module = importlib.import_module(f"optexity_private.portals.{portal}.methods")
        fn = getattr(module, function_name)
        if function_name.startswith("_") or not inspect.iscoroutinefunction(fn):
            raise ValueError(f"'{function_name}' is not a dispatchable function")

        await start_task_in_server(task)
        memory = Memory(unique_child_arn=unique_child_arn)

        params = inspect.signature(fn).parameters
        kwargs = {}
        if "processed_cookie_data" in params:
            processed_cookie_data = await module.get_processed_cookie_data_in_server(
                task
            )
            if not processed_cookie_data:
                raise ValueError(
                    f"No saved session cookies found for portal '{portal}' "
                    f"(task {task.task_id}); cannot call '{function_name}'"
                )
            kwargs["processed_cookie_data"] = processed_cookie_data
        if "task" in params:
            kwargs["task"] = task
        if "memory" in params:
            kwargs["memory"] = memory

        # Dispatchable functions are real async I/O (httpx, not requests) and
        # return (success, message) rather than raising for expected failures,
        # so this can be awaited directly without blocking the event loop
        # this shares with /health and /kill_task for every other task.
        success, message = await fn(**kwargs)
        if success:
            task.status = "success"
            logger.info(
                f"Marketplace function succeeded for task {task.task_id}: {message}"
            )
        else:
            task.status = "failed"
            task.error = message
            logger.error(
                f"Marketplace function failed for task {task.task_id}: {message}"
            )
    except Exception as e:
        logger.error(
            f"Marketplace function call failed for task {task.task_id} "
            f"({task.endpoint_name}): {e}"
        )
        task.error = str(e)
        task.status = "failed"

    task.completed_at = datetime.now(timezone.utc)
    if memory is not None:
        await save_output_data_in_server(task, memory)
        await save_downloads_in_server(task, memory)
    try:
        await complete_task_in_server(
            task,
            memory.token_usage if memory else None,
            child_process_id,
            unique_child_arn,
        )
        await initiate_callback(task)
    except Exception as fail_err:
        logger.error(f"Failed to report task {task.task_id} completion: {fail_err}")


async def task_processor():
    """Background worker that processes tasks from the queue one at a time."""
    global task_running, last_task_start_time, current_task_timeout_minutes
    logger.info("Task processor started")

    while True:
        try:
            *_, task = await task_queue.get()
            if task.task_id in tasks_to_kill:
                logger.info(f"Task {task.task_id} has been killed")
                tasks_to_kill.remove(task.task_id)
                continue

            marketplace = parse_marketplace_endpoint_name(task.endpoint_name)

            if marketplace is not None and marketplace[0] == "fn":
                task_running = True
                last_task_start_time = datetime.now(timezone.utc)
                current_task_timeout_minutes = task.max_timeout_in_minutes
                await run_marketplace_function(task, unique_child_arn, child_process_id)
                continue

            automation_error: str | None = None
            if task.endpoint_name.startswith(MARKETPLACE_ENDPOINT_PREFIX):
                # Marketplace automations are plain .py modules shipped in
                # optexity_private (worker image only) — built directly instead
                # of fetching a recording-backed automation from opcloud.
                fetch_success = False
                try:
                    if marketplace is None:
                        raise ValueError(
                            "expected marketplace_<portal>_fn_<function> or "
                            "marketplace_<portal>_automation_<workflow>"
                        )
                    _, portal, automation_name = marketplace
                    module = importlib.import_module(
                        f"optexity_private.portals.{portal}.{automation_name}"
                    )
                    task.automation = module.automation
                    fetch_success = True
                    logger.info(
                        f"Loaded marketplace automation '{automation_name}' "
                        f"(portal={portal}) for task {task.task_id}"
                    )
                except (ImportError, AttributeError, ValueError) as marketplace_err:
                    automation_error = (
                        f"Failed to load marketplace automation "
                        f"'{task.endpoint_name}': {marketplace_err}"
                    )
                    logger.error(f"{automation_error} (task {task.task_id})")
            else:
                # Fetch fresh automation from server just before running so any
                # workflow changes after allocation are picked up. Client errors
                # (<500) retry 3x with 3s wait; 5xx / unreachable use up to 4 min
                # exponential backoff.
                recording_url = settings.GET_RECORDING_ENDPOINT.format(
                    recording_id=task.recording_id
                )
                fetch_url = f"{settings.SERVER_URL.rstrip('/')}/{recording_url}"
                fetch_success = False
                response, attempt = await request_with_backoff(
                    fetch_url,
                    headers={"x-api-key": task.api_key},
                    log_label=f"automation fetch for task {task.task_id}",
                )
                if response is not None:
                    try:
                        data = response.json()
                        task.automation = Automation.model_validate(data["automation"])
                        # Use recording/workspace callback_url only if no per-task
                        # override exists on either field (task_callback_url takes
                        # priority; task.callback_url may have been set via x-callback-url
                        # header and must not be overwritten).
                        if (
                            task.callback_url is None
                            and not task.task_callback_url
                            and data.get("callback_url")
                        ):
                            from optexity.schema.task import CallbackUrl

                            try:
                                task.callback_url = CallbackUrl.model_validate(
                                    data["callback_url"]
                                )
                            except Exception as cb_err:
                                logger.warning(
                                    f"Failed to parse callback_url for task "
                                    f"{task.task_id}: {cb_err}"
                                )
                        fetch_success = True
                        logger.info(
                            f"Fetched fresh automation for task {task.task_id} "
                            f"(recording {task.recording_id})"
                        )
                    except Exception as parse_err:
                        logger.warning(
                            f"Failed to parse automation response for task "
                            f"{task.task_id}: {parse_err}"
                        )
                if not fetch_success:
                    automation_error = (
                        f"Failed to fetch automation after {attempt} attempts"
                    )

            if not fetch_success:
                if task.automation is not None:
                    logger.warning(
                        f"All automation fetch attempts failed for task {task.task_id}; "
                        f"using in-memory fallback"
                    )
                else:
                    logger.error(
                        f"All automation fetch attempts failed for task {task.task_id}; "
                        f"marking failed and firing callback"
                    )
                    task.status = "failed"
                    task.error = automation_error
                    task.completed_at = datetime.now(timezone.utc)
                    try:
                        await complete_task_in_server(
                            task, None, child_process_id, unique_child_arn
                        )
                        await initiate_callback(task)
                    except Exception as fail_err:
                        logger.error(
                            f"Failed to report task {task.task_id} failure: {fail_err}"
                        )
                    continue

            task_running = True
            last_task_start_time = datetime.now(timezone.utc)
            current_task_timeout_minutes = task.max_timeout_in_minutes
            await run_automation_in_process(task, unique_child_arn, child_process_id)

        except asyncio.CancelledError:
            logger.info("Task processor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
        finally:
            task_running = False
            last_task_start_time = None
            current_task_timeout_minutes = None


async def register_with_master():
    global unique_child_arn
    """Register with master on startup (handles restarts automatically)."""
    # Get my task metadata from ECS
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("http://169.254.170.2/v3/task")
        response.raise_for_status()
        metadata = response.json()

    logger.info(f"Metadata from ECS: {metadata}")
    my_task_arn = metadata["TaskARN"]
    unique_child_arn = str(my_task_arn)
    my_ip = metadata["Containers"][0]["Networks"][0]["IPv4Addresses"][0]

    my_port = None
    my_stream_port = None
    for binding in metadata["Containers"][0].get("NetworkBindings", []):
        if binding["containerPort"] == settings.CHILD_PORT_OFFSET:
            my_port = binding["hostPort"]
        elif binding["containerPort"] == settings.WEBSOCKIFY_PORT:
            my_stream_port = binding["hostPort"]

    if not my_port:
        logger.error("Could not find host port binding")
        raise ValueError("Host port not found in metadata")

    if not my_stream_port:
        logger.error("Could not find stream port binding")
        raise ValueError("Stream port not found in metadata")

    # Register with master
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"http://{settings.SERVER_URL}/register_child",
            json={
                "task_arn": my_task_arn,
                "private_ip": my_ip,
                "port": my_port,
                "stream_port": my_stream_port,
            },
        )
        response.raise_for_status()

    logger.info(f"Registered with master: {response.json()}")


def get_app_with_endpoints(is_aws: bool, child_id: int, port: int = -1):
    global child_process_id, _child_fastapi_port
    child_process_id = child_id
    _child_fastapi_port = port

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _global_actual_browser
        """Lifespan context manager for startup and shutdown."""
        # Startup

        if is_aws:
            try:
                await register_with_master()
                logger.info("Registered with master")
            except Exception:
                logger.exception(
                    "Failed to register with master, using fallback UUID as child ARN"
                )
        else:
            logger.info("Not running on AWS, skipping master registration")

        asyncio.create_task(task_processor())
        logger.info("Task processor background task started")
        yield
        # Shutdown (if needed in the future)
        logger.info("Shutting down task processor")

        if _global_actual_browser is not None:
            logger.debug("Stopping actual browser on lifecycle end")
            await _global_actual_browser.stop(graceful=True)
            _global_actual_browser = None
            logger.debug("Actual browser stopped on lifecycle end")

        logger.info("Lifecycle ended")

    app = FastAPI(title="Optexity Inference", lifespan=lifespan)

    @app.get("/is_task_running", tags=["info"])
    async def is_task_running():
        """Is task running endpoint."""
        return task_running

    @app.post("/human_in_loop_completed")
    async def human_in_loop_completed_child(body: HumanInLoopCompletedBody = Body(...)):
        """Called by opcloud when the human has finished the HITL step."""
        hitl_completed_tasks.add(body.task_id)
        return JSONResponse({"success": True})

    @app.get("/hitl_status")
    async def hitl_status(task_id: str):
        """Polled by the worker subprocess every 5 seconds during HITL pause."""
        completed = task_id in hitl_completed_tasks
        if completed:
            hitl_completed_tasks.discard(task_id)
        return {"completed": completed}

    @app.post("/kill_task")
    async def kill_task(task_ids: list[str] = Body(...)):
        """Kill task endpoint (bulk).

        Accepts a list of task IDs. For each:
        - Adds to tasks_to_kill so queued tasks are skipped by the processor
          and a running worker's post-exit retry loop bails out.
        - SIGKILLs the worker subprocess if currently running.
        """
        for task_id in task_ids:
            tasks_to_kill.add(task_id)
            hitl_completed_tasks.discard(task_id)
            proc = running_task_processes.get(task_id)
            if proc is not None and proc.returncode is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    logger.info(f"Killed worker process group for task {task_id}")
                except ProcessLookupError:
                    logger.info(
                        f"Worker process group for task {task_id} was already gone; "
                        "treating /kill_task as successful"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to kill worker process for task {task_id}: {e}"
                    )
        return JSONResponse(
            content={
                "success": True,
                "message": f"Kill signal sent for {len(task_ids)} tasks",
            },
            status_code=200,
        )

    @app.get("/health", tags=["info"])
    async def health():
        """Health check endpoint.

        Returns 503 if a task has been running longer than its
        ``max_timeout_in_minutes`` (default 15). Valid long runs stay healthy
        until that task-specific limit.
        """
        timeout_minutes = current_task_timeout_minutes or 15
        if (
            task_running
            and last_task_start_time is not None
            and datetime.now(timezone.utc) - last_task_start_time
            > timedelta(minutes=timeout_minutes)
        ):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": (f"Task not finished within {timeout_minutes} minutes"),
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "task_running": task_running,
                "queued_tasks": task_queue.qsize(),
            },
        )

    @app.post("/set_child_process_id", tags=["info"])
    async def set_child_process_id(request: ChildProcessIdRequest):
        """Set child process id endpoint."""
        global child_process_id, unique_child_arn
        child_process_id = int(request.new_child_process_id)
        unique_child_arn = request.new_unique_child_arn
        return JSONResponse(
            content={"success": True, "message": "Child process id has been set"},
            status_code=200,
        )

    @app.post("/allocate_task")
    async def allocate_task(tasks: list[Task] = Body(...)):
        """Bulk allocate tasks onto this child's local priority queue."""
        try:
            for task in tasks:
                _enqueue_task(task)
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"{len(tasks)} task(s) allocated. Check status at https://dashboard.optexity.com/tasks",
                },
                status_code=202,
            )
        except Exception as e:
            logger.error(f"Error allocating tasks: {e}")
            return JSONResponse(
                content={"success": False, "message": str(e)}, status_code=500
            )

    if not is_aws:

        @app.post("/inference")
        async def inference(inference_request: InferenceRequest = Body(...)):
            response_data: dict | None = None
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = urljoin(settings.SERVER_URL, settings.INFERENCE_ENDPOINT)
                    headers = {"x-api-key": settings.OPTEXITY_API_KEY}
                    response = await client.post(
                        url, json=inference_request.model_dump(), headers=headers
                    )
                    response_data = response.json()
                    response.raise_for_status()

                assert response_data is not None
                task_data = response_data["task"]

                task = Task.model_validate_json(task_data)
                if task.use_proxy and settings.PROXY_URL is None:
                    raise ValueError(
                        "PROXY_URL is not set and is required when use_proxy is True"
                    )
                task.is_dedicated = inference_request.is_dedicated
                task.allocated_at = datetime.now(timezone.utc)
                _enqueue_task(task)

                return JSONResponse(
                    content={
                        "success": True,
                        "message": "Task has been allocated. Check its status and output at https://dashboard.optexity.com/tasks",
                        "task_id": task.task_id,
                    },
                    status_code=202,
                )

            except Exception as e:
                error = str(e)
                if response_data is not None:
                    error = response_data.get("error", str(e))

                logger.error(f"❌ Error fetching recordings: {error}")
                return JSONResponse({"success": False, "error": error}, status_code=500)

    return app


def main():
    """Main function to run the server."""
    parser = argparse.ArgumentParser(
        description="Dynamic API endpoint generator for Optexity recordings"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the server ",
    )
    parser.add_argument(
        "--child_process_id",
        type=int,
        help="Child process ID",
    )
    parser.add_argument(
        "--is_aws",
        action="store_true",
        help="Is child process",
        default=False,
    )

    args = parser.parse_args()

    app = get_app_with_endpoints(
        is_aws=args.is_aws, child_id=args.child_process_id, port=args.port
    )

    # Start the server (this is blocking and manages its own event loop)
    logger.info(f"Starting server on {args.host}:{args.port}")
    run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
