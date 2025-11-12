#!/usr/bin/env python3
"""
Dynamic API endpoint generator for Optexity recordings.

This script:
1. Takes an API key from command line arguments
2. Fetches all recordings from localhost:8000/api/v1/get_recordings
3. Dynamically creates FastAPI endpoints for each recording's endpoint_name
4. Each endpoint executes the automation from the recording
"""

import argparse
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn import run

from optexity.inference.core.run_automation import run_automation
from optexity.inference.infra.browser import Browser
from optexity.schema.automation import Automation
from optexity.schema.memory import Memory, Variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    asyncio.create_task(task_processor())
    logger.info("Task processor background task started")
    yield
    # Shutdown (if needed in the future)
    logger.info("Shutting down task processor")


app = FastAPI(title="Dynamic Optexity Endpoints", lifespan=lifespan)


class TaskStatus(str, Enum):
    """Task status enumeration."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a queued automation task."""

    task_id: str
    endpoint_name: str
    automation: Automation
    input_variables: Dict[str, Any]
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancelled: bool = False


class RecordingResponse(BaseModel):
    endpoint_name: str | None
    automation: Automation | None
    description: str | None = None


class TaskRequest(BaseModel):
    """Request body for dynamic endpoint tasks."""

    input_variables: Dict[str, list[str]] = {}


# Store recordings and their automations
recordings_cache: Dict[str, RecordingResponse] = {}

# Task queue and storage
task_queue: asyncio.Queue[Task] = asyncio.Queue()
tasks: Dict[str, Task] = {}
task_processor_running = False


async def fetch_recordings(
    api_key: str, base_url: str = "http://localhost:8000"
) -> list[RecordingResponse]:
    """Fetch all recordings from the API."""
    url = f"{base_url}/api/v1/get_recordings"
    headers = {"X-API-Key": api_key}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            recordings = [
                RecordingResponse.model_validate(recording)
                for recording in response.json()
            ]
            logger.info(f"Fetched {len(recordings)} recordings")
            return recordings
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error fetching recordings: {e.response.status_code} - {e.response.text}"
        )
        raise
    except Exception as e:
        logger.error(f"Error fetching recordings: {e}")
        raise


async def execute_automation_task(task: Task):
    """Execute an automation task."""
    # Check if task was cancelled before starting
    if task.cancelled:
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        return

    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now()

    # Convert input_variables to the format expected by Memory
    formatted_variables = {
        key: [value] if not isinstance(value, list) else value
        for key, value in task.input_variables.items()
    }

    memory = Memory(variables=Variables(input_variables=formatted_variables))
    browser = Browser(headless=False)

    try:
        await browser.start()

        # Check if cancelled after browser start
        if task.cancelled:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            await browser.stop()
            return

        # Navigate to the automation URL if provided
        if task.automation.url:
            await browser.go_to_url(task.automation.url)
            await asyncio.sleep(2)  # Wait for page to load

        # Check if cancelled before running automation
        if task.cancelled:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            await browser.stop()
            return

        # Run the automation
        await run_automation(task.automation, memory, browser)

        # Store the results
        task.result = {
            "success": True,
            "output_data": memory.variables.output_data,
            "generated_variables": memory.variables.generated_variables,
            "browser_states_count": len(memory.browser_states),
        }
        task.status = TaskStatus.COMPLETED

    except Exception as e:
        logger.error(f"Error executing automation task {task.task_id}: {e}")
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.result = {
            "success": False,
            "error": str(e),
        }
    finally:
        task.completed_at = datetime.now()
        await browser.stop()


async def task_processor():
    """Background worker that processes tasks from the queue one at a time."""
    global task_processor_running
    task_processor_running = True
    logger.info("Task processor started")

    while True:
        task = None
        try:
            # Get next task from queue (blocks until one is available)
            task = await task_queue.get()

            # Check if task was cancelled while in queue
            if task.cancelled:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                task_queue.task_done()
                continue

            logger.info(
                f"Processing task {task.task_id} for endpoint {task.endpoint_name}"
            )
            await execute_automation_task(task)
            logger.info(
                f"Task {task.task_id} completed with status {task.status.value}"
            )

            task_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Task processor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
            if task:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
            if task:
                task_queue.task_done()

    task_processor_running = False


def create_dynamic_endpoint(
    endpoint_name: str, automation: Automation, description: str = None
):
    """Create a dynamic FastAPI endpoint for a recording."""

    # Sanitize endpoint name for URL
    sanitized_name = endpoint_name.replace(" ", "_").replace("/", "_").lower()
    endpoint_path = f"/{sanitized_name}"

    @app.post(endpoint_path, tags=["dynamic_endpoints"])
    async def dynamic_endpoint(body: TaskRequest):
        """Dynamically created endpoint for automation execution."""
        try:
            # Extract input variables from request body
            input_variables: dict = body.input_variables

            # Create a new task
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                endpoint_name=endpoint_name,
                automation=automation,
                input_variables=input_variables,
            )

            # Store task
            tasks[task_id] = task

            # Queue the task
            await task_queue.put(task)
            logger.info(f"Queued task {task_id} for endpoint {endpoint_name}")

            # Return immediately with task ID
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Task has been queued",
                    "task_id": task_id,
                    "status": task.status.value,
                },
                status_code=202,  # Accepted
            )

        except Exception as e:
            logger.error(f"Error in dynamic endpoint {endpoint_path}: {e}")
            return JSONResponse(
                content={"success": False, "error": str(e)}, status_code=500
            )

    # Also create a GET endpoint for info
    @app.get(f"{endpoint_path}/info", tags=["dynamic_endpoints"])
    async def endpoint_info():
        """Get information about this endpoint."""
        return {
            "endpoint_name": endpoint_name,
            "description": description,
            "url": automation.url if automation else None,
            "parameters": automation.parameters if automation else [],
        }

    logger.info(f"Created endpoint: POST {endpoint_path}")


async def setup_endpoints(api_key: str, base_url: str = "http://localhost:8000"):
    """Fetch recordings and create dynamic endpoints."""
    recordings = await fetch_recordings(api_key, base_url)

    for recording in recordings:

        if not recording.endpoint_name:
            logger.warning(
                f"Skipping recording without endpoint_name: {recording.model_dump_json()}"
            )
            continue

        if not recording.automation:
            logger.warning(
                f"Skipping recording {recording.endpoint_name} without automation data"
            )
            continue

        try:
            # Store in cache
            recordings_cache[recording.endpoint_name] = recording

            # Create dynamic endpoint
            create_dynamic_endpoint(
                recording.endpoint_name, recording.automation, recording.description
            )

        except Exception as e:
            logger.error(f"Error processing recording {recording.endpoint_name}: {e}")
            continue

    logger.info(f"Setup complete. Created {len(recordings_cache)} endpoints")


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with information about available endpoints."""
    endpoints = []
    for endpoint_name, data in recordings_cache.items():
        sanitized = endpoint_name.replace(" ", "_").replace("/", "_").lower()
        endpoints.append(
            {
                "name": endpoint_name,
                "endpoint": f"/{sanitized}",
                "description": data.description,
            }
        )

    return {
        "message": "Dynamic Optexity API",
        "total_endpoints": len(endpoints),
        "endpoints": endpoints,
    }


@app.get("/health", tags=["info"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "endpoints_count": len(recordings_cache),
        "task_processor_running": task_processor_running,
        "queued_tasks": task_queue.qsize(),
    }


@app.get("/tasks", tags=["tasks"])
async def list_tasks(status: Optional[str] = None):
    """List all tasks, optionally filtered by status."""
    task_list = []
    for task_id, task in tasks.items():
        if status is None or task.status.value == status:
            task_list.append(
                {
                    "task_id": task.task_id,
                    "endpoint_name": task.endpoint_name,
                    "status": task.status.value,
                    "created_at": (
                        task.created_at.isoformat() if task.created_at else None
                    ),
                    "started_at": (
                        task.started_at.isoformat() if task.started_at else None
                    ),
                    "completed_at": (
                        task.completed_at.isoformat() if task.completed_at else None
                    ),
                    "error": task.error,
                }
            )

    # Sort by created_at (newest first)
    task_list.sort(key=lambda x: x["created_at"] or "", reverse=True)

    return {
        "total": len(task_list),
        "tasks": task_list,
    }


@app.get("/tasks/{task_id}", tags=["tasks"])
async def get_task(task_id: str):
    """Get details of a specific task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    response = {
        "task_id": task.task_id,
        "endpoint_name": task.endpoint_name,
        "status": task.status.value,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "cancelled": task.cancelled,
    }

    # Include result if task is completed or failed
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        response["result"] = task.result

    if task.error:
        response["error"] = task.error

    return response


@app.delete("/tasks/{task_id}", tags=["tasks"])
async def cancel_task(task_id: str):
    """Cancel a queued or running task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    # Only allow cancellation of queued or running tasks
    if task.status not in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
        return JSONResponse(
            content={
                "success": False,
                "message": f"Cannot cancel task with status {task.status.value}",
            },
            status_code=400,
        )

    # Mark task as cancelled
    task.cancelled = True
    if task.status == TaskStatus.QUEUED:
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()

    logger.info(f"Cancelled task {task_id}")

    return JSONResponse(
        content={
            "success": True,
            "message": "Task cancelled",
            "task_id": task_id,
            "status": task.status.value,
        },
        status_code=200,
    )


async def setup_app(api_key: str, base_url: str):
    """Setup the application by fetching recordings and creating endpoints."""
    logger.info("Setting up dynamic endpoints...")
    await setup_endpoints(api_key, base_url)

    if len(recordings_cache) == 0:
        logger.warning("No recordings found. Server will start with no endpoints.")


def main():
    """Main function to run the server."""
    parser = argparse.ArgumentParser(
        description="Dynamic API endpoint generator for Optexity recordings"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for authenticating with the recordings API",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the recordings API (default: http://localhost:8000)",
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
        default=8001,
        help="Port to run the server on (default: 8001)",
    )

    args = parser.parse_args()

    # Setup endpoints before starting server (run in async context)
    asyncio.run(setup_app(args.api_key, args.base_url))

    # Start the server (this is blocking and manages its own event loop)
    logger.info(f"Starting server on {args.host}:{args.port}")
    run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
