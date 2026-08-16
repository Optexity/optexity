from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import KeyPressAction
from optexity.schema.memory import Memory


async def handle_key_press(
    keypress_action: KeyPressAction,
    memory: Memory,
    browser: Browser,
):
    if browser.backend_agent is None:
        raise RuntimeError(
            "Cannot send keys: Browser Use action runtime is unavailable"
        )

    keys = keypress_action.keys
    if keys is None:
        if isinstance(keypress_action.type, list):
            keys = "+".join(keypress_action.type)
        else:
            keys = keypress_action.type
    if not keys:
        raise RuntimeError("Cannot send keys: no key sequence was provided")

    action_model = browser.backend_agent.ActionModel(
        send_keys={"keys": keys},  # pyright: ignore[reportCallIssue]
    )
    results = await browser.backend_agent.multi_act([action_model])
    if not results:
        raise RuntimeError("Browser Use returned no result for send_keys")
    if results[0].error:
        raise RuntimeError(f"Browser Use send_keys failed: {results[0].error}")
