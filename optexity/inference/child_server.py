import argparse
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from uvicorn import run

from optexity.inference.core.run_automation import run_automation
from optexity.schema.task import Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

child_process_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    asyncio.create_task(task_processor())
    logger.info("Task processor background task started")
    yield
    # Shutdown (if needed in the future)
    logger.info("Shutting down task processor")


app = FastAPI(title="Optexity Inference", lifespan=lifespan)
task_running = False
task_queue: asyncio.Queue[Task] = asyncio.Queue()


async def task_processor():
    """Background worker that processes tasks from the queue one at a time."""
    global task_running
    logger.info("Task processor started")

    while True:
        try:
            # Get next task from queue (blocks until one is available)
            task = await task_queue.get()
            task_running = True
            await run_automation(task, child_process_id)
            task_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Task processor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
        finally:

            task_running = False


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


@app.get("/is_task_running", tags=["info"])
async def is_task_running():
    """Is task running endpoint."""
    return task_running


@app.get("/health", tags=["info"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "task_running": task_running,
        "queued_tasks": task_queue.qsize(),
    }


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
        "--child_process_id",
        type=int,
        help="Number of child processes to run",
    )

    args = parser.parse_args()

    assert args.child_process_id is not None
    assert args.child_process_id >= 0
    assert args.child_process_id < 10

    global child_process_id
    child_process_id = args.child_process_id

    port = 9000 + args.child_process_id
    # Start the server (this is blocking and manages its own event loop)
    logger.info(f"Starting server on {args.host}:{port}")
    run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
