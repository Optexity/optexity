import asyncio
import base64
import logging

import pyautogui
import pyperclip

from optexity.schema.actions.powershell_action import PowerShellAction

logger = logging.getLogger(__name__)


def _encode_powershell(script: str) -> str:
    """Base64-encode a PowerShell script for use with -EncodedCommand."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


async def run_powershell_action(powershell_action: PowerShellAction):
    """Open PowerShell on the RDP session (machine 2), run commands, and close it."""
    logger.debug(
        f"Running PowerShell action with {len(powershell_action.commands)} command(s)"
    )
    # Join all commands into a single script and run via -EncodedCommand
    full_script = "\n".join(powershell_action.commands)
    encoded = _encode_powershell(full_script)
    run_cmd = f"powershell -EncodedCommand {encoded}"

    logger.debug(f"Executing encoded PowerShell script ({len(full_script)} chars)")
    pyperclip.copy(run_cmd)
    await asyncio.sleep(0.2)
    pyautogui.hotkey("ctrl", "v")
    await asyncio.sleep(0.3)
    pyautogui.press("enter")
    await asyncio.sleep(1)

    # Close PowerShell
    pyperclip.copy("exit")
    await asyncio.sleep(0.2)
    pyautogui.hotkey("ctrl", "v")
    await asyncio.sleep(0.3)
    pyautogui.press("enter")

    logger.debug("PowerShell action completed successfully")
