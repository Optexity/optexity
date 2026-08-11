"""Runtime context handed to ``python_script`` nodes that ask for it.

Script nodes can compute a file's bytes but historically had no way to say
"this is a downloadable output of the task" — only ``expect_download``
interaction nodes could. That forced authors to base64 the bytes back into
the page, build a ``Blob`` + ``<a download>`` anchor, and add a second
``click_element`` node purely so Chromium would emit a download event.

``ScriptContext.save_download`` closes that gap. It joins the download model
at the point every existing path already converges: write into
``task.downloads_directory``, append to ``memory.downloads``, and register
metadata into ``memory.download_metadata`` using the same
``resolve_download_metadata_template`` helper ``handle_download`` uses. The
capture mechanics of ``expect_download`` are untouched.
"""

import asyncio
import inspect
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import aiofiles

from optexity.exceptions import ExpectedDownloadFailedException
from optexity.inference.infra.browser import Browser
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.utils.utils import resolve_download_metadata_template

logger = logging.getLogger(__name__)

# Filesystem-hostile characters and control chars. Mirrors the sanitization
# rules that automation prep scripts have been reimplementing by hand.
_UNSAFE_FILENAME_CHARS = re.compile(r'[/\\:*?"\'<>|\x00-\x1f]')
_WHITESPACE_RUN = re.compile(r"\s+")
_MAX_FILENAME_LENGTH = 150
# Guard against an unbounded rename loop on a pathological directory.
_MAX_DEDUPE_ATTEMPTS = 1000


def sanitize_download_filename(
    filename: str, max_length: int = _MAX_FILENAME_LENGTH
) -> str:
    """Make a user-visible label safe to use as a filename.

    Strips path separators and control characters, collapses whitespace runs,
    drops trailing dots/spaces, and truncates while preserving the extension.
    """
    name = _UNSAFE_FILENAME_CHARS.sub("_", str(filename))
    name = _WHITESPACE_RUN.sub(" ", name).strip()
    name = name.strip(". ")

    if not name:
        raise ValueError(f"filename is empty after sanitization: {filename!r}")

    if len(name) > max_length:
        suffix = Path(name).suffix
        # A "suffix" longer than the budget is not a real extension; drop it.
        if len(suffix) >= max_length:
            suffix = ""
        stem = name[: len(name) - len(suffix)] if suffix else name
        name = stem[: max_length - len(suffix)].strip(". ") + suffix

    return name


def _unique_path(directory: Path, filename: str) -> Path:
    """Return a path in ``directory`` that does not collide with an existing file.

    Appends ``_2``, ``_3``, ... to the stem, matching the de-duplication
    convention already used by automation prep scripts.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for counter in range(2, _MAX_DEDUPE_ATTEMPTS + 2):
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate

    raise ValueError(
        f"could not find a free filename for {filename!r} in {directory} "
        f"after {_MAX_DEDUPE_ATTEMPTS} attempts"
    )


class ScriptContext:
    """Optional third argument for ``python_script`` node functions.

    Opt in by naming it in the script's signature::

        async def code_fn(axtree, browser, ctx):
            await ctx.save_download("report.csv", csv_bytes,
                                    metadata={"kind": "export"})
            return {"saved": 1}

    Scripts that keep the original ``code_fn(axtree, browser)`` /
    ``code_fn(page)`` signatures never receive a context and are unaffected.
    """

    def __init__(self, task: Task, memory: Memory, browser: Browser | None = None):
        self.task = task
        self.memory = memory
        self.browser = browser

    # ---- downloads ----

    async def save_download(
        self,
        filename: str,
        content: bytes | str | None = None,
        *,
        path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Register a file as a downloadable output of this task.

        Provide exactly one of ``content`` (bytes or text held in memory) or
        ``path`` (a file already on disk, which is moved rather than copied).

        Unlike ``expect_download`` actions, ``filename`` is required: the script
        author already knows the name, so there is no Chromium-supplied name to
        reconcile and no reason to fall back to a UUID.

        Returns the final path, which may differ from ``filename`` if it needed
        sanitizing or de-duplication.
        """
        if (content is None) == (path is None):
            raise ValueError(
                "save_download requires exactly one of content= or path= "
                f"(got content={'set' if content is not None else 'None'}, "
                f"path={'set' if path is not None else 'None'})"
            )

        safe_name = sanitize_download_filename(filename)
        downloads_directory = self.task.downloads_directory
        downloads_directory.mkdir(parents=True, exist_ok=True)
        download_path = _unique_path(downloads_directory, safe_name)

        try:
            if content is not None:
                data = (
                    content.encode("utf-8")
                    if isinstance(content, str)
                    else bytes(content)
                )
                async with aiofiles.open(download_path, "wb") as f:
                    await f.write(data)
            else:
                source = Path(path)  # type: ignore[arg-type]
                if not source.is_file():
                    raise ExpectedDownloadFailedException(
                        f"save_download source path does not exist: {source}"
                    )
                await asyncio.to_thread(shutil.move, str(source), str(download_path))

            if not (download_path.exists() and download_path.stat().st_size > 0):
                raise ExpectedDownloadFailedException(
                    f"save_download produced an empty or missing file: {download_path}"
                )
        except Exception:
            # save_downloads_in_server uploads whatever it finds in the
            # downloads directory, so a partial or empty file left behind here
            # would ship to S3 as a bogus artifact.
            download_path.unlink(missing_ok=True)
            raise

        self.memory.downloads.append(download_path)
        self._register_download_metadata(download_path.name, metadata)

        logger.info(
            f"save_download: saved {download_path.name!r} "
            f"({download_path.stat().st_size} bytes) to {downloads_directory}"
        )
        return download_path

    def _register_download_metadata(
        self, filename: str, metadata: dict[str, Any] | None
    ) -> None:
        """Same resolution semantics as ``handle_download``'s metadata hook."""
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise ValueError(
                f"save_download metadata must be a dict, got {type(metadata).__name__}"
            )
        try:
            resolved = resolve_download_metadata_template(
                metadata,
                self.task.input_parameters,
                self.memory.variables.generated_variables,
                self.task.unique_parameters or {},
            )
            self.memory.download_metadata[filename] = resolved or {}
            logger.info(
                f"save_download: registered metadata for {filename!r}: {resolved}"
            )
        except Exception as e:
            logger.warning(
                f"save_download: failed to register metadata for {filename!r}: {e}"
            )

    @property
    def downloads_dir(self) -> Path:
        return self.task.downloads_directory

    # ---- cross-node state ----

    @property
    def state(self) -> dict[str, Any]:
        """Plain dict shared by every script node in this run.

        Each script node is ``exec``'d with fresh globals, so module-level
        variables do not survive between nodes. Use this instead of stashing
        work lists on ``window`` — it costs no JS round trip and survives
        navigation.
        """
        return self.memory.state

    # ---- read-only views ----

    @property
    def variables(self) -> dict:
        """Variables produced by earlier nodes, before template substitution."""
        return self.memory.variables.generated_variables

    @property
    def input_parameters(self) -> dict:
        return self.task.input_parameters

    @property
    def unique_parameters(self) -> dict:
        return self.task.unique_parameters or {}

    # ---- convenience ----

    async def get_page(self):
        """The live Playwright page. Raises if this context has no browser."""
        if self.browser is None:
            raise ValueError("ScriptContext has no browser attached")
        return await self.browser.get_current_page()

    def log(self, message: Any, level: str = "info") -> None:
        """Log through the run's logger so diagnostics land in the task logs.

        Tags the message with the current step index so lines from different
        script nodes (or different loop iterations of the same node) can be
        told apart in the task-wide log file. ``level`` is one of the
        standard logging level names (``"debug"``, ``"info"``, ``"warning"``,
        ``"error"``).
        """
        log_fn = getattr(logger, level.lower(), None)
        if not callable(log_fn):
            raise ValueError(f"ScriptContext.log: unknown level {level!r}")
        step = self.memory.automation_state.step_index
        log_fn(f"[python_script step={step}] {message}")


# A script opts into the context by naming the parameter, not by arity. Matching
# on position instead would silently hand the context to an unrelated third
# parameter in an existing script.
_CONTEXT_PARAM_NAMES = ("ctx", "context")


async def call_script_fn(code_fn: Callable, args: tuple, ctx: ScriptContext):
    """Await ``code_fn(*args)``, adding ``ctx`` only if it asks for it by name.

    Scripts using the historical signatures — ``code_fn(axtree, browser)`` for
    extraction and ``code_fn(page)`` for interaction — are called exactly as
    before.
    """
    try:
        signature = inspect.signature(code_fn)
    except (TypeError, ValueError):
        # Builtins / C callables have no introspectable signature.
        return await code_fn(*args)

    param = next(
        (
            signature.parameters[name]
            for name in _CONTEXT_PARAM_NAMES
            if name in signature.parameters
        ),
        None,
    )

    if param is None:
        return await code_fn(*args)

    if param.kind is inspect.Parameter.VAR_KEYWORD:
        # `**ctx` is a catch-all, not a request for the context.
        return await code_fn(*args)

    if param.kind is inspect.Parameter.POSITIONAL_ONLY:
        return await code_fn(*args, ctx)

    return await code_fn(*args, **{param.name: ctx})
