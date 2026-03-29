import pyautogui

from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    KeyCombinationAction,
    KeyPressAction,
    KeyPressType,
)
from optexity.schema.memory import Memory


async def handle_key_press(
    keypress_action: KeyPressAction,
    memory: Memory,
    browser: Browser,
):

    if browser.channel == "rdp" or browser.backend == "computer-vision":
        await handle_keypress_native(keypress_action, memory, browser)
        return

    page = await browser.get_current_page()
    if page is None:
        return

    if keypress_action.type == KeyPressType.ENTER:
        await page.keyboard.press("Enter")
    if keypress_action.type == KeyPressType.TAB:
        await page.keyboard.press("Tab")
    if keypress_action.type == KeyPressType.ZERO:
        await page.keyboard.press("0")
    if keypress_action.type == KeyPressType.ONE:
        await page.keyboard.press("1")
    if keypress_action.type == KeyPressType.TWO:
        await page.keyboard.press("2")
    if keypress_action.type == KeyPressType.THREE:
        await page.keyboard.press("3")
    if keypress_action.type == KeyPressType.FOUR:
        await page.keyboard.press("4")
    if keypress_action.type == KeyPressType.FIVE:
        await page.keyboard.press("5")
    if keypress_action.type == KeyPressType.SIX:
        await page.keyboard.press("6")
    if keypress_action.type == KeyPressType.SEVEN:
        await page.keyboard.press("7")
    if keypress_action.type == KeyPressType.EIGHT:
        await page.keyboard.press("8")
    if keypress_action.type == KeyPressType.NINE:
        await page.keyboard.press("9")
    if keypress_action.type == KeyPressType.SLASH:
        await page.keyboard.press("/")
    if keypress_action.type == KeyPressType.SPACE:
        await page.keyboard.press("Space")


async def handle_keypress_native(
    keypress_action: KeyPressAction | KeyCombinationAction,
    memory: Memory,
    browser: Browser,
):
    if isinstance(keypress_action, KeyPressAction):
        pyautogui.press(keypress_action.type.value)

    elif isinstance(keypress_action, KeyCombinationAction):
        pyautogui.press(
            [
                key.value
                for key in keypress_action.key_combination
                if isinstance(key, KeyPressType)
            ]
        )
