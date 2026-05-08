import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class ScreenshotRecorder:
    """Records a remote CDP-accessible browser by periodically taking screenshots
    and encoding them into an mp4 via ffmpeg's image2pipe input.

    Used for browser-use (cloud) where x11grab is unavailable since the browser
    renders remotely and there is no local X display to capture.

    Lifecycle:
        recorder = ScreenshotRecorder(output_dir, cdp_url)
        await recorder.start()
        ...
        path = await recorder.stop()
    """

    def __init__(
        self,
        output_dir: Path,
        cdp_url: str,
        filename: str = "recording.mp4",
        fps: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.cdp_url = cdp_url
        self.fps = fps
        self.output_path: Path = self.output_dir / filename
        self._stop_event: asyncio.Event = asyncio.Event()
        self._capture_task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None

    def is_running(self) -> bool:
        return self._capture_task is not None and not self._capture_task.done()

    async def start(self) -> None:
        if shutil.which("ffmpeg") is None:
            logger.error(
                "[recording] ffmpeg not found in PATH — ScreenshotRecorder disabled"
            )
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info(
                f"[recording] ScreenshotRecorder ffmpeg started pid={self._proc.pid} "
                f"output={self.output_path}"
            )
        except Exception as e:
            logger.error(f"[recording] ScreenshotRecorder failed to start ffmpeg: {e}")
            self._proc = None
            return

        self._capture_task = asyncio.create_task(self._capture_loop())

    async def _capture_loop(self) -> None:
        from playwright.async_api import async_playwright

        interval = 1.0 / self.fps
        playwright_inst = None

        try:
            playwright_inst = await async_playwright().start()
            browser = await playwright_inst.chromium.connect_over_cdp(self.cdp_url)
            logger.info(
                f"[recording] ScreenshotRecorder connected to CDP: {self.cdp_url}"
            )

            while not self._stop_event.is_set():
                try:
                    contexts = browser.contexts
                    if not contexts:
                        await asyncio.sleep(interval)
                        continue

                    pages = contexts[0].pages
                    if not pages:
                        await asyncio.sleep(interval)
                        continue

                    page = pages[0]
                    screenshot_bytes = await asyncio.wait_for(
                        page.screenshot(type="jpeg", quality=70),
                        timeout=3.0,
                    )

                    if (
                        self._proc is not None
                        and self._proc.stdin is not None
                        and not self._proc.stdin.is_closing()
                        and self._proc.returncode is None
                    ):
                        self._proc.stdin.write(screenshot_bytes)
                        await self._proc.stdin.drain()

                except asyncio.TimeoutError:
                    logger.debug("[recording] screenshot timeout — skipping frame")
                except Exception as e:
                    logger.debug(f"[recording] screenshot error (non-fatal): {e}")

                await asyncio.sleep(interval)

        except Exception as e:
            logger.error(f"[recording] ScreenshotRecorder capture loop error: {e}")
        finally:
            if playwright_inst is not None:
                try:
                    await playwright_inst.stop()
                except BaseException:
                    # BaseException covers CancelledError so cleanup runs even
                    # when the capture task is cancelled via asyncio.wait_for timeout.
                    pass
            logger.info("[recording] ScreenshotRecorder capture loop finished")

    async def stop(self, timeout: float = 15.0) -> Path | None:
        self._stop_event.set()

        if self._capture_task is not None and not self._capture_task.done():
            try:
                await asyncio.wait_for(self._capture_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    pass
            self._capture_task = None

        if self._proc is None:
            return self.output_path if self.output_path.exists() else None

        if self._proc.returncode is not None:
            self._proc = None
            return self.output_path if self.output_path.exists() else None

        try:
            if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass

            try:
                await asyncio.wait_for(self._proc.wait(), timeout=timeout)
                rc = self._proc.returncode
                if rc != 0:
                    stderr_bytes = b""
                    if self._proc.stderr:
                        try:
                            stderr_bytes = await asyncio.wait_for(
                                self._proc.stderr.read(), timeout=2.0
                            )
                        except Exception:
                            pass
                    logger.error(
                        f"[recording] ScreenshotRecorder ffmpeg exited with rc={rc}: "
                        f"{stderr_bytes.decode(errors='replace')[:500]}"
                    )
                else:
                    logger.info(
                        f"[recording] ScreenshotRecorder ffmpeg exited gracefully rc={rc}"
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "[recording] ScreenshotRecorder ffmpeg timeout — killing"
                )
                try:
                    self._proc.kill()
                    await self._proc.wait()
                except Exception:
                    pass
        finally:
            self._proc = None

        if self.output_path.exists():
            size = self.output_path.stat().st_size
            if size == 0:
                logger.error(
                    f"[recording] ScreenshotRecorder output is zero bytes — recording failed: {self.output_path}"
                )
                return None
            logger.info(
                f"[recording] ScreenshotRecorder output: {self.output_path} ({size} bytes)"
            )
            return self.output_path

        logger.error(
            f"[recording] ScreenshotRecorder output missing: {self.output_path}"
        )
        return None

    def get_video_path(self) -> Path | None:
        if self.output_path.exists():
            return self.output_path
        return None
