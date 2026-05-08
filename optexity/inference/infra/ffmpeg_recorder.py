import asyncio
import logging
import os
import platform
import shutil
import signal
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegRecorder:
    """Records the X11 display to an mp4 via ffmpeg's x11grab.

    Replaces Playwright's `record_video_dir`. Captures whatever is on
    `$DISPLAY`, so it works for any browser (chromium, chrome, cloakbrowser,
    native CDP) that renders to the local X server. Skipped for `browser-use`
    since that browser runs in the cloud.

    Lifecycle:
        recorder = FFmpegRecorder(output_dir)
        await recorder.start()      # spawn ffmpeg
        ...                         # automation happens
        path = await recorder.stop()  # graceful shutdown finalises mp4 moov atom
    """

    def __init__(
        self,
        output_dir: Path,
        filename: str = "recording.mp4",
        display: str | None = None,
        fps: int = 10,
        resolution: str = "1920x1080",
    ):
        self.output_dir = Path(output_dir)
        self.display = display or os.environ.get("DISPLAY", ":99")
        self.fps = fps
        self.resolution = resolution
        self.proc: asyncio.subprocess.Process | None = None
        self.output_path: Path = self.output_dir / filename

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        if self.is_running():
            logger.warning(
                "[recording] FFmpegRecorder.start called but ffmpeg is already running"
            )
            return

        if platform.system() != "Linux":
            logger.warning(
                f"[recording] FFmpegRecorder only supports Linux/x11grab; "
                f"current platform={platform.system()} — recording skipped"
            )
            return

        if shutil.which("ffmpeg") is None:
            logger.error(
                "[recording] ffmpeg binary not found in PATH — recording disabled"
            )
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "x11grab",
            "-framerate",
            str(self.fps),
            "-video_size",
            self.resolution,
            "-i",
            self.display,
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

        logger.info(f"[recording] Starting ffmpeg: {' '.join(cmd)}")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            logger.info(
                f"[recording] ffmpeg started pid={self.proc.pid} "
                f"output={self.output_path}"
            )
        except Exception as e:
            logger.error(f"[recording] failed to start ffmpeg: {e}")
            self.proc = None

    async def stop(self, timeout: float = 10.0) -> Path | None:
        if self.proc is None:
            logger.info("[recording] FFmpegRecorder.stop: no ffmpeg process to stop")
            return self.output_path if self.output_path.exists() else None

        if self.proc.returncode is not None:
            logger.info(f"[recording] ffmpeg already exited rc={self.proc.returncode}")
            self.proc = None
            return self.output_path if self.output_path.exists() else None

        try:
            # Send 'q' to stdin for a clean shutdown (writes the mp4 moov atom).
            if self.proc.stdin is not None and not self.proc.stdin.is_closing():
                try:
                    self.proc.stdin.write(b"q")
                    await self.proc.stdin.drain()
                    self.proc.stdin.close()
                except Exception as e:
                    logger.warning(
                        f"[recording] failed to send 'q' to ffmpeg stdin: {e}"
                    )

            try:
                await asyncio.wait_for(self.proc.wait(), timeout=timeout)
                rc = self.proc.returncode
                if rc != 0:
                    stderr_bytes = b""
                    if self.proc.stderr:
                        try:
                            stderr_bytes = await asyncio.wait_for(
                                self.proc.stderr.read(), timeout=2.0
                            )
                        except Exception:
                            pass
                    logger.error(
                        f"[recording] ffmpeg exited with rc={rc}: "
                        f"{stderr_bytes.decode(errors='replace')[:500]}"
                    )
                else:
                    logger.info(f"[recording] ffmpeg exited gracefully rc={rc}")
            except asyncio.TimeoutError:
                logger.warning(
                    f"[recording] ffmpeg did not exit within {timeout}s — sending SIGTERM"
                )
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    await asyncio.wait_for(self.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "[recording] ffmpeg unresponsive after SIGTERM — sending SIGKILL"
                    )
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"[recording] error terminating ffmpeg: {e}")
        finally:
            self.proc = None

        if self.output_path.exists():
            size = self.output_path.stat().st_size
            if size == 0:
                logger.error(
                    f"[recording] ffmpeg output is zero bytes — recording failed: {self.output_path}"
                )
                return None
            logger.info(f"[recording] ffmpeg output: {self.output_path} ({size} bytes)")
            return self.output_path
        logger.error(f"[recording] ffmpeg output missing: {self.output_path}")
        return None

    def get_video_path(self) -> Path | None:
        if self.output_path.exists():
            return self.output_path
        return None
