# Optexity CLI
#
# Usage:
#   optexity <command> [options]
#
# Commands:
#
#   install-browsers
#     Install required Playwright/Patchright browsers.
#     Example:
#       optexity install-browsers
#
#   inference
#     Run the Optexity inference server.
#     Options:
#       --host HOST                 Bind address (default: 0.0.0.0)
#       --port PORT                 Port (default: 9000)
#       --child-process-id ID       Child process ID (default: 0)
#       --is-aws                    Enable AWS mode (registers with master)
#     Example:
#       optexity inference --port 9001 --child-process-id 1
#
#   record
#     Open a browser or RDP session and record user actions.
#     One of --browser or --rdp-host is required.
#
#     Browser mode:
#       --browser chrome|chromium   Browser channel to launch
#       --url URL                   Navigate to URL on start (optional)
#       --output-dir DIR            Directory for recordings (default: recordings)
#     Example:
#       optexity record --browser chrome
#       optexity record --browser chrome --url https://example.com
#
#     RDP mode:
#       --rdp-host HOST             RDP host to connect to
#       --rdp-username USER         RDP username
#       --rdp-password PASS         RDP password
#       --output-dir DIR            Directory for recordings (default: recordings)
#     Example:
#       optexity record --rdp-host 192.168.1.100 --rdp-username admin --rdp-password secret
#
#     Recording controls:
#       Press Enter                 Start recording after browser opens
#       ESC / F12 / Ctrl+Shift+Q   Stop recording and run post-processing
#       Ctrl+C                      Stop without post-processing
#                                   (set RECORDER_POST_ON_CTRL_C=1 to enable post-processing on Ctrl+C)

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import threading

from dotenv import load_dotenv
from uvicorn import run

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")
else:
    load_dotenv(env_path)


def install_browsers() -> None:
    """Install Playwright + Patchright browsers."""
    try:
        subprocess.run(
            ["playwright", "install", "--with-deps", "chromium", "chrome"],
            check=True,
        )
        subprocess.run(
            ["patchright", "install", "chromium", "chrome"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install browsers", exc_info=True)
        sys.exit(e.returncode)


async def _run_record(args: argparse.Namespace) -> None:
    import uuid

    from optexity.inference.infra.actual_browser import ActualBrowser
    from optexity.recorder import Recorder

    if args.rdp_host:
        from optexity.schema.automation import RDPParameter

        rdp_parameter = RDPParameter(
            host=args.rdp_host,
            username=args.rdp_username,
            password=args.rdp_password,
        )
        browser = ActualBrowser(
            channel="rdp",
            unique_child_arn=str(uuid.uuid4()),
            rdp_parameter=rdp_parameter,
        )
    else:
        browser = ActualBrowser(
            channel=args.browser,
            unique_child_arn=str(uuid.uuid4()),
        )

    await browser.start()

    if args.url:
        await browser.goto_url(args.url)

    try:
        logger.info(
            "Browser started.\n"
            "  Stop recording : ESC, F12, or Ctrl+Shift+Q\n"
            "  Ctrl+C         : stops without post-processing (set RECORDER_POST_ON_CTRL_C=1 to enable)\n"
            "Press Enter to begin recording..."
        )
        await asyncio.get_event_loop().run_in_executor(None, input)

        recorder = Recorder(output_dir=args.output_dir)
        thread = threading.Thread(target=recorder.start, daemon=True)
        thread.start()
        try:
            await asyncio.get_event_loop().run_in_executor(None, thread.join)
        except asyncio.CancelledError:
            recorder.stop()
            thread.join(timeout=10)  # wait for flush/cleanup before post-processing
            recorder._run_post_process_gui_recording()
            raise
    finally:
        await browser.stop()


def run_record(args: argparse.Namespace) -> None:
    asyncio.run(_run_record(args))


def run_inference(args: argparse.Namespace) -> None:
    from optexity.inference.child_process import get_app_with_endpoints

    app = get_app_with_endpoints(is_aws=args.is_aws, child_id=args.child_process_id)
    run(
        app,
        host=args.host,
        port=args.port,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="optexity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # install-browsers
    # ---------------------------
    install_cmd = subparsers.add_parser(
        "install_browsers",
        help="Install required browsers for Optexity",
        aliases=["install-browsers"],
    )
    install_cmd.set_defaults(func=lambda _: install_browsers())

    # ---------------------------
    # inference
    # ---------------------------
    inference_cmd = subparsers.add_parser(
        "inference", help="Run Optexity inference server"
    )
    inference_cmd.add_argument("--host", default="0.0.0.0")
    inference_cmd.add_argument("--port", type=int, default=9000)
    inference_cmd.add_argument(
        "--child_process_id", "--child-process-id", type=int, default=0
    )
    inference_cmd.add_argument(
        "--is_aws", "--is-aws", action="store_true", default=False
    )

    inference_cmd.set_defaults(func=run_inference)

    # ---------------------------
    # record
    # ---------------------------
    record_cmd = subparsers.add_parser(
        "record",
        help="Open a browser and record user actions",
    )
    browser_group = record_cmd.add_mutually_exclusive_group(required=True)
    browser_group.add_argument(
        "--browser",
        choices=["chrome", "chromium"],
        help="Browser channel to open (with optional --url)",
    )
    browser_group.add_argument(
        "--rdp-host",
        "--rdp_host",
        dest="rdp_host",
        help="RDP host to connect to",
    )
    record_cmd.add_argument("--url", default=None, help="URL to open in browser")
    record_cmd.add_argument(
        "--rdp-username", "--rdp_username", dest="rdp_username", default=None
    )
    record_cmd.add_argument(
        "--rdp-password", "--rdp_password", dest="rdp_password", default=None
    )
    record_cmd.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", default="recordings"
    )
    record_cmd.set_defaults(func=run_record)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
