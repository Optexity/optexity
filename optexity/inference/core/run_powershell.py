import asyncio
import logging
import subprocess

import pyautogui

from optexity.schema.actions.powershell_action import PowerShellAction

logger = logging.getLogger(__name__)


def _xdotool_type(text: str) -> None:
    """Type text into the focused X11 window using xdotool. No clipboard needed."""
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "20", text], check=True
    )


async def run_powershell_action(powershell_action: PowerShellAction):
    """Open PowerShell on the RDP session (machine 2), run commands, and close it."""
    logger.debug(
        f"Running PowerShell action with {len(powershell_action.commands)} command(s)"
    )
    # Execute each command
    for cmd in powershell_action.commands:
        logger.debug(f"Executing command: {cmd}")
        _xdotool_type(cmd)
        await asyncio.sleep(0.3)
        pyautogui.press("enter")
        await asyncio.sleep(1)

    # Close PowerShell
    _xdotool_type("exit")
    await asyncio.sleep(0.3)
    pyautogui.press("enter")

    logger.debug("PowerShell action completed successfully")
