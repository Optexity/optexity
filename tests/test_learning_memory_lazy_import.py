from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class LearningMemoryLazyImportTests(unittest.TestCase):
    def test_normal_runtime_import_does_not_require_history_compiler(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["LEARNING_MEMORY_ENABLED"] = "false"

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import optexity.inference.core.run_automation; print('ok')",
            ],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
