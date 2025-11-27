import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

import httpx
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn import run

from optexity.inference.core.run_automation import run_automation
from optexity.schema.inference import InferenceRequest
from optexity.schema.task import Task
from optexity.utils.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChildProcessIdRequest(BaseModel):
    new_child_process_id: str


child_process_id = None
task_running = False
last_task_start_time = None
task_queue: asyncio.Queue[Task] = asyncio.Queue()


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
            await run_automation(task, child_process_id)

        except asyncio.CancelledError:
            logger.info("Task processor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
        finally:

            task_running = False


async def register_with_master():
    """Register with master on startup (handles restarts automatically)."""
    # Get my task metadata from ECS
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("http://169.254.170.2/v3/task")
        response.raise_for_status()
        metadata = response.json()

    my_task_arn = metadata["TaskARN"]
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


def get_app_with_endpoints(is_aws: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
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

    if not is_aws:

        @app.post("/inference")
        async def inference(inference_request: InferenceRequest = Body(...)):
            try:

                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = urljoin(settings.SERVER_URL, settings.INFERENCE_ENDPOINT)
                    headers = {"x-api-key": settings.API_KEY}
                    response = await client.post(
                        url, json=inference_request.model_dump(), headers=headers
                    )
                    response.raise_for_status()

                response_data = response.json()
                task_data = response_data["task"]

                task = Task.model_validate_json(task_data)
                task.allocated_at = datetime.now(timezone.utc)
                await task_queue.put(task)

                return JSONResponse(
                    content={"success": True, "message": "Task has been allocated"},
                    status_code=202,
                )

            except Exception as e:
                logger.error(f"❌ Error fetching recordings: {str(e)}")
                return JSONResponse(
                    {"success": False, "error": str(e)}, status_code=500
                )

    if is_aws:

        @app.post("/allocate_task")
        async def allocate_task(task: Task = Body(...)):
            """Get details of a specific task."""
            try:

                await task_queue.put(task)
                return JSONResponse(
                    content={"success": True, "message": "Task has been allocated"},
                    status_code=202,
                )
            except Exception as e:
                logger.error(f"Error allocating task {task.task_id}: {e}")
                return JSONResponse(
                    content={"success": False, "message": str(e)}, status_code=500
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

    app = get_app_with_endpoints(args.is_aws)

    global child_process_id
    child_process_id = args.child_process_id

    # Start the server (this is blocking and manages its own event loop)
    logger.info(f"Starting server on {args.host}:{args.port}")
    run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
