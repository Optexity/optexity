from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import KeyPressAction, KeyPressType
from optexity.schema.memory import Memory


async def handle_key_press(
    keypress_action: KeyPressAction,
    memory: Memory,
    browser: Browser,
):
    page = await browser.get_current_page()
    if page is None:
        return

    key_type = keypress_action.type
    if isinstance(key_type, str):
        key_type = key_type.capitalize()

    if key_type == KeyPressType.ENTER:
        await page.keyboard.press("Enter")
    if key_type == KeyPressType.TAB:
        await page.keyboard.press("Tab")
    if key_type == KeyPressType.ZERO:
        await page.keyboard.press("0")
    if key_type == KeyPressType.ONE:
        await page.keyboard.press("1")
    if key_type == KeyPressType.TWO:
        await page.keyboard.press("2")
    if key_type == KeyPressType.THREE:
        await page.keyboard.press("3")
    if key_type == KeyPressType.FOUR:
        await page.keyboard.press("4")
    if key_type == KeyPressType.FIVE:
        await page.keyboard.press("5")
    if key_type == KeyPressType.SIX:
        await page.keyboard.press("6")
    if key_type == KeyPressType.SEVEN:
        await page.keyboard.press("7")
    if key_type == KeyPressType.EIGHT:
        await page.keyboard.press("8")
    if key_type == KeyPressType.NINE:
        await page.keyboard.press("9")
    if key_type == KeyPressType.SLASH:
        await page.keyboard.press("/")
    if key_type == KeyPressType.SPACE:
        await page.keyboard.press("Space")
