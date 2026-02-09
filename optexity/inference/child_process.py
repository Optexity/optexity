import argparse
import asyncio
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
from typing import TextIO
from urllib.parse import urljoin

import httpx
import psutil
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn import run

from optexity.inference.infra.actual_browser import ActualBrowser
from optexity.schema.inference import InferenceRequest
from optexity.schema.memory import SystemInfo
from optexity.schema.task import Task
from optexity.utils.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChildProcessIdRequest(BaseModel):
    new_child_process_id: str


child_process_id = -1
unique_child_arn: str = str(uuid.uuid4())
task_running = False
last_task_start_time = None
task_queue: asyncio.Queue[Task] = asyncio.Queue()
_global_actual_browser: ActualBrowser | None = None


def log_system_info(f: TextIO):
    f.write(
        json.dumps(
            {
                "container_memory_total": SystemInfo().total_system_memory / 1024,
                "container_memory_used": SystemInfo().total_system_memory_used / 1024,
                "percent_container_memory_used": round(
                    SystemInfo().total_system_memory_used
                    / SystemInfo().total_system_memory,
                    2,
                ),
            }
        )
        + "\n"
    )
    f.write(
        json.dumps(
            {
                "host_memory_total": psutil.virtual_memory().total / (1024**3),
                "host_memory_used": psutil.virtual_memory().used / (1024**3),
                "percent_host_memory_used": round(
                    psutil.virtual_memory().used / psutil.virtual_memory().total, 2
                ),
            }
        )
        + "\n",
    )


async def run_automation_in_process(
    task: Task, unique_child_arn: str, child_process_id: int
):
    with open("/tmp/system_info.json", "a") as f:
        f.write("=" * 100 + "\n")
        f.write("----- System info for Task " + task.task_id + ": -------\n")
        f.write("Before starting browser\n")
        log_system_info(f)

    global _global_actual_browser

    if not task.is_dedicated and _global_actual_browser is not None:
        await _global_actual_browser.stop(graceful=False)
        _global_actual_browser = None

    if _global_actual_browser is None:
        _global_actual_browser = ActualBrowser(
            channel="chrome",
            unique_child_arn=unique_child_arn,
            port=9222 + child_process_id,
            headless=False,
            is_dedicated=True,
        )
        await _global_actual_browser.start()

    with open("/tmp/system_info.json", "a") as f:
        f.write("After starting browser\n")
        log_system_info(f)

    logger.debug("Running automation in process")
    worker_path = pathlib.Path(__file__).parent / "worker.py"
    proc = subprocess.Popen(
        [
            sys.executable,
            worker_path,
            task.model_dump_json(),
            unique_child_arn,
            str(child_process_id),
        ],
        preexec_fn=os.setsid,  # isolate process group
    )

    try:
        logger.debug("Waiting for automation to finish")
        returncode = proc.wait(timeout=600)  # seconds
        logger.debug("Automation finished in process")
        return returncode
    except subprocess.TimeoutExpired:
        logger.debug("Automation timed out in process")
        os.killpg(proc.pid, signal.SIGKILL)
        logger.debug("Automation killed in process")
        return -1
    finally:
        with open("/tmp/system_info.json", "a") as f:
            f.write("After automation finished in process\n")
            log_system_info(f)

        if _global_actual_browser is not None and not task.is_dedicated:
            logger.debug("Stopping actual browser as not dedicated")
            try:
                await _global_actual_browser.stop(graceful=True)
                _global_actual_browser = None
            except Exception as e:
                logger.error(f"Error stopping actual browser: {e}")

        with open("/tmp/system_info.json", "a") as f:
            f.write("After stopping actual browser\n")
            log_system_info(f)


async def task_processor():
    """Background worker that processes tasks from the queue one at a time."""
    global task_running
    global last_task_start_time
    logger.info("Task processor started")

    while True:
        try:
            # Get next task from queue (blocks until one is available)
            task = await task_queue.get()
            task_running = True
            last_task_start_time = datetime.now()
            await run_automation_in_process(task, unique_child_arn, child_process_id)

        except asyncio.CancelledError:
            logger.info("Task processor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
        finally:

            task_running = False


async def register_with_master():
    global unique_child_arn
    """Register with master on startup (handles restarts automatically)."""
    # Get my task metadata from ECS
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("http://169.254.170.2/v3/task")
        response.raise_for_status()
        metadata = response.json()

    my_task_arn = metadata["TaskARN"]
    unique_child_arn = str(my_task_arn)
    my_ip = metadata["Containers"][0]["Networks"][0]["IPv4Addresses"][0]

    my_port = None
    for binding in metadata["Containers"][0].get("NetworkBindings", []):
        if binding["containerPort"] == settings.CHILD_PORT_OFFSET:
            my_port = binding["hostPort"]
            break

    if not my_port:
        logger.error("Could not find host port binding")
        raise ValueError("Host port not found in metadata")

    # Register with master
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"http://{settings.SERVER_URL}/register_child",
            json={"task_arn": my_task_arn, "private_ip": my_ip, "port": my_port},
        )
        response.raise_for_status()

    logger.info(f"Registered with master: {response.json()}")


def get_app_with_endpoints(is_aws: bool, child_id: int):
    global child_process_id
    child_process_id = child_id

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _global_actual_browser
        """Lifespan context manager for startup and shutdown."""
        # Startup

        if is_aws:
            asyncio.create_task(register_with_master())

        logger.info("Registered with master")
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

    @app.get("/health", tags=["info"])
    async def health():
        """Health check endpoint."""
        global last_task_start_time
        if (
            task_running
            and last_task_start_time
            and datetime.now() - last_task_start_time > timedelta(minutes=15)
        ):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Task not finished in the last 15 minutes",
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
        global child_process_id
        child_process_id = int(request.new_child_process_id)
        return JSONResponse(
            content={"success": True, "message": "Child process id has been set"},
            status_code=200,
        )

    @app.post("/allocate_task")
    async def allocate_task(task: Task = Body(...)):
        """Get details of a specific task."""
        try:

            await task_queue.put(task)
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Task has been allocated. Check its status and output at https://dashboard.optexity.com/tasks",
                },
                status_code=202,
            )
        except Exception as e:
            logger.error(f"Error allocating task {task.task_id}: {e}")
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
                    headers = {"x-api-key": settings.API_KEY}
                    response = await client.post(
                        url, json=inference_request.model_dump(), headers=headers
                    )
                    response_data = response.json()
                    response.raise_for_status()

                task_data = response_data["task"]

                task = Task.model_validate_json(task_data)
                if task.use_proxy and settings.PROXY_URL is None:
                    raise ValueError(
                        "PROXY_URL is not set and is required when use_proxy is True"
                    )
                task.is_dedicated = inference_request.is_dedicated
                task.allocated_at = datetime.now(timezone.utc)
                await task_queue.put(task)

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

    app = get_app_with_endpoints(is_aws=args.is_aws, child_id=args.child_process_id)

    # Start the server (this is blocking and manages its own event loop)
    logger.info(f"Starting server on {args.host}:{args.port}")
    run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
