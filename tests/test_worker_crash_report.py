import os
import subprocess
import sys
from pathlib import Path

from optexity.inference import worker
from optexity.inference.child_process import _read_crash_error
from optexity.schema.enums import WORKER_ERROR_FILE_ENV, ExitCodes


def test_worker_writes_startup_crash_for_parent(tmp_path: Path):
    error_file = tmp_path / "worker_error.txt"
    proc = subprocess.run(
        [sys.executable, worker.__file__, '{"task_id": "t1"}', "arn", "0", "cdp"],
        env={**os.environ, WORKER_ERROR_FILE_ENV: str(error_file)},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == ExitCodes.WORKER_CRASHED.value
    assert "Traceback" in proc.stderr
    error = _read_crash_error(error_file, proc.returncode)
    assert error.startswith("ValidationError: ")
    assert len(error) <= worker.WORKER_ERROR_MAX_CHARS


def test_read_crash_error_falls_back_to_exit_code(tmp_path: Path):
    assert (
        _read_crash_error(tmp_path / "missing.txt", 1)
        == "Worker crashed with exit code 1"
    )
