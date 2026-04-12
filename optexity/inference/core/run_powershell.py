import asyncio
import logging

import pyautogui
import pyperclip

from optexity.schema.actions.powershell_action import PowerShellAction

logger = logging.getLogger(__name__)


async def run_powershell_action(powershell_action: PowerShellAction):
    """Open PowerShell on the RDP session (machine 2), run commands, and close it."""
    logger.debug(
        f"Running PowerShell action with {len(powershell_action.commands)} command(s)"
    )
    # Execute each command
    for cmd in powershell_action.commands:
        logger.debug(f"Executing command: {cmd}")
        pyperclip.copy(cmd)
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
