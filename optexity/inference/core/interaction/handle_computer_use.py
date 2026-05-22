"""Computer-use agentic task runner.

Drives a multi-turn loop with a provider's Computer Use tool (Claude today;
Gemini intended to slot in later). Each turn:

    1. Send the running conversation (with the latest screenshot) to the model.
    2. Execute every ``tool_use`` block the model emitted, using pyautogui /
       the existing Browser screenshot helper.
    3. Append ``tool_result`` entries for each executed action and loop.

The loop exits cleanly on ``stop_reason == "end_turn"`` or when ``max_steps``
is reached (logs a warning, does not raise).

Constraints honoured:
    * RDP only (enforced upstream by ``validate_rdp_parameter``).
    * No deterministic fallbacks — the model decides every action.
    * No wall-clock timeout — ``max_steps`` is the only fuse.
    * API key sourced from ``ANTHROPIC_API_KEY`` via the existing model layer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

import pyautogui

from optexity.inference.infra.browser import Browser
from optexity.inference.models import get_llm_model
from optexity.inference.models.anthropic import Anthropic
from optexity.inference.models.llm_model import AnthropicModels
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory, OutputData
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_COMPUTER_USE_MODEL = AnthropicModels.CLAUDE_SONNET_4_6

# Claude's `key` action uses xdotool key names; pyautogui has its own set.
# Translate the common ones — anything not in this map is passed through
# (pyautogui accepts most ASCII keys verbatim).
_CLAUDE_TO_PYAUTOGUI_KEY = {
    "Return": "enter",
    "Escape": "esc",
    "BackSpace": "backspace",
    "Tab": "tab",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Home": "home",
    "End": "end",
    "Page_Up": "pageup",
    "Page_Down": "pagedown",
    "Delete": "delete",
    "Insert": "insert",
    "Print": "printscreen",
    "super": "win",
    "Super_L": "win",
    "Super_R": "win",
    "control": "ctrl",
    "Control_L": "ctrl",
    "Control_R": "ctrl",
    "Shift_L": "shift",
    "Shift_R": "shift",
    "Alt_L": "alt",
    "Alt_R": "alt",
    "space": "space",
}


async def run_computer_use_agent(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
) -> None:
    """Public entry point for Computer Use on RDP.

    Claude-only for now. When Gemini support lands, this is where the
    provider dispatch will go (driven by ``agentic_task.model``).
    """
    await _run_claude_loop(agentic_task_action, task, memory, browser)


# ---------------------------------------------------------------------------
# Claude loop
# ---------------------------------------------------------------------------


async def _run_claude_loop(
    action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
) -> None:
    model = _resolve_claude_model()
    step_directory = task.logs_directory / f"step_{memory.automation_state.step_index}"
    step_directory.mkdir(parents=True, exist_ok=True)

    system_instruction = (
        "You are controlling a remote desktop via the `computer` tool. "
        "Execute the user's task using mouse, keyboard, and screenshots. "
        "Prefer graceful interactions (e.g. click the close button, accept save "
        "dialogs) over forcing actions. When the task is complete, stop "
        "calling tools and reply with a short summary."
    )

    initial_screenshot = await browser.get_screenshot()
    if not initial_screenshot:
        logger.error("computer_use: failed to capture initial screenshot")
        return

    _save_screenshot_b64(initial_screenshot, step_directory / "initial_screenshot.png")

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": action.task},
                _image_block(initial_screenshot),
            ],
        }
    ]

    final_text_summary: str | None = None

    for turn in range(action.max_steps):
        turn_directory = step_directory / f"computer_use_turn_{turn}"
        turn_directory.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"computer_use: turn {turn + 1}/{action.max_steps} "
            f"(model={model.model_name.value})"
        )

        try:
            content_blocks, stop_reason, token_usage = await asyncio.to_thread(
                model.run_computer_use_turn,
                messages,
                system_instruction,
            )
        except Exception as e:
            logger.error(f"computer_use: model call failed on turn {turn}: {e}")
            return

        memory.token_usage += token_usage

        # Log the raw assistant response for debugging.
        _write_json(turn_directory / "response.json", _serialize_blocks(content_blocks))

        # Append the assistant turn to the conversation.
        messages.append(
            {"role": "assistant", "content": _serialize_blocks(content_blocks)}
        )

        # Collect any text the model emitted this turn — overwrite so we keep
        # the most recent summary; on end_turn this becomes the final one.
        turn_text = _join_text_blocks(content_blocks)
        if turn_text:
            final_text_summary = turn_text

        # Execute every tool_use the model emitted (sequentially) and gather
        # their tool_results for the next user message.
        tool_use_blocks = [b for b in content_blocks if _block_type(b) == "tool_use"]
        if not tool_use_blocks:
            # No actions to dispatch — Claude is either done (end_turn) or
            # waiting on something we can't provide. Either way, stop.
            logger.debug(
                f"computer_use: no tool_use blocks (stop_reason={stop_reason}); "
                "exiting loop"
            )
            break

        dispatched: list[dict] = []
        tool_results: list[dict] = []
        for block in tool_use_blocks:
            action_input = _block_input(block) or {}
            tool_use_id = _block_id(block)
            action_name = action_input.get("action", "")
            logger.debug(
                f"computer_use: dispatching action='{action_name}' "
                f"args={ {k: v for k, v in action_input.items() if k != 'action'} }"
            )

            content_for_result, is_error = await _dispatch_claude_action(
                action_input, browser
            )

            dispatched.append(
                {
                    "tool_use_id": tool_use_id,
                    "action": action_name,
                    "input": action_input,
                    "is_error": is_error,
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content_for_result,
                    "is_error": is_error,
                }
            )

        _write_json(turn_directory / "dispatched_actions.json", dispatched)

        # Persist the screenshots we sent back to Claude as tool_results.
        # Pure diagnostic — lets us open the PNG and visually compare against
        # the coords Claude returned next turn ("did the click land where the
        # model intended? was the model targeting the right button?").
        for idx, tr in enumerate(tool_results):
            content = tr.get("content")
            if isinstance(content, list):
                for sub in content:
                    if isinstance(sub, dict) and sub.get("type") == "image":
                        b64 = sub.get("source", {}).get("data")
                        if b64:
                            _save_screenshot_b64(
                                b64, turn_directory / f"post_action_{idx}.png"
                            )
                            break

        messages.append({"role": "user", "content": tool_results})

        if stop_reason == "end_turn":
            logger.debug("computer_use: stop_reason=end_turn — task complete")
            break
    else:
        logger.warning(
            f"computer_use: reached max_steps={action.max_steps} without "
            "end_turn — exiting (task may be incomplete)"
        )

    if final_text_summary:
        memory.variables.output_data.append(
            OutputData(
                unique_identifier="computer_use_summary",
                text=final_text_summary,
            )
        )


def _resolve_claude_model() -> Anthropic:
    """Instantiate the default Claude Computer Use model via the existing layer."""
    model = get_llm_model(
        DEFAULT_CLAUDE_COMPUTER_USE_MODEL, use_structured_output=False
    )
    if not isinstance(model, Anthropic):
        raise RuntimeError(
            f"computer_use: resolved model is not Anthropic: {type(model).__name__}"
        )
    return model


# ---------------------------------------------------------------------------
# Dispatch — Claude action → pyautogui / Browser
# ---------------------------------------------------------------------------


# Actions that don't need a fresh post-action screenshot:
#   * ``screenshot`` already returns an image, capturing again would be wasteful.
#   * ``cursor_position`` is an informational query; the model wants the coords
#     back as text, not a new frame.
# Every other action mutates the visual state, so we follow Anthropic's
# reference Computer Use loop and return a fresh screenshot as the tool_result
# so the model can see what its action produced.
_SKIP_POST_ACTION_SCREENSHOT = {"screenshot", "cursor_position"}


async def _dispatch_claude_action(
    action_input: dict,
    browser: Browser,
) -> tuple[Any, bool]:
    """Execute one Claude `computer` tool action.

    After a successful state-changing action, the returned content is replaced
    by a fresh screenshot so the model sees the post-action state (matches
    Anthropic's reference Computer Use loop). Errors and informational actions
    (``screenshot``, ``cursor_position``) pass through unchanged.

    Returns
    -------
    tuple[Any, bool]
        (content_for_tool_result, is_error). Content is either a list[image_block]
        or a string, depending on the action and outcome.
    """
    action_name = action_input.get("action")
    if not action_name:
        return "Missing 'action' field", True

    handler = _ACTION_HANDLERS.get(action_name)
    if handler is None:
        return f"Unknown action: {action_name}", True

    try:
        content, is_error = await handler(action_input, browser)
    except Exception as e:
        logger.warning(f"computer_use: action '{action_name}' failed: {e}")
        return f"Action '{action_name}' failed: {e}", True

    if not is_error and action_name not in _SKIP_POST_ACTION_SCREENSHOT:
        try:
            content = await _capture_image_block(browser)
        except Exception as e:
            # Don't fail the action just because the post-action screenshot
            # didn't land — fall back to the handler's original ack content
            # so the model still gets a meaningful tool_result.
            logger.warning(
                f"computer_use: post-action screenshot for '{action_name}' failed: {e}"
            )

    return content, is_error


# Each handler returns (content_for_tool_result, is_error).
# Synchronous pyautogui calls are wrapped in asyncio.to_thread so the event
# loop stays responsive.


async def _h_screenshot(_input: dict, browser: Browser) -> tuple[Any, bool]:
    return await _capture_image_block(browser), False


async def _h_mouse_move(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    x, y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.moveTo, x, y)
    return f"moved to ({x}, {y})", False


async def _h_left_click(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    x, y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.click, x, y)
    return f"left-clicked ({x}, {y})", False


async def _h_right_click(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    x, y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.click, x, y, button="right")
    return f"right-clicked ({x}, {y})", False


async def _h_middle_click(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    x, y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.click, x, y, button="middle")
    return f"middle-clicked ({x}, {y})", False


async def _h_double_click(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    x, y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.doubleClick, x, y)
    return f"double-clicked ({x}, {y})", False


async def _h_triple_click(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    x, y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.tripleClick, x, y)
    return f"triple-clicked ({x}, {y})", False


async def _h_left_click_drag(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    start_x, start_y = _coord(action_input, "start_coordinate")
    end_x, end_y = _coord(action_input, "coordinate")
    await asyncio.to_thread(pyautogui.moveTo, start_x, start_y)
    await asyncio.to_thread(pyautogui.dragTo, end_x, end_y, button="left")
    return f"dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})", False


async def _h_left_mouse_down(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    coord = action_input.get("coordinate")
    if coord:
        x, y = int(coord[0]), int(coord[1])
        await asyncio.to_thread(pyautogui.moveTo, x, y)
    await asyncio.to_thread(pyautogui.mouseDown, button="left")
    return "left mouse down", False


async def _h_left_mouse_up(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    coord = action_input.get("coordinate")
    if coord:
        x, y = int(coord[0]), int(coord[1])
        await asyncio.to_thread(pyautogui.moveTo, x, y)
    await asyncio.to_thread(pyautogui.mouseUp, button="left")
    return "left mouse up", False


async def _h_type(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    text = action_input.get("text", "")
    if not isinstance(text, str):
        return "type: 'text' must be a string", True
    await asyncio.to_thread(pyautogui.typewrite, text, 0.05)
    return f"typed {len(text)} chars", False


async def _h_key(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    combo = action_input.get("text", "")
    if not isinstance(combo, str) or not combo:
        return "key: 'text' must be a non-empty string", True
    keys = [_translate_key(k) for k in combo.split("+")]
    await asyncio.to_thread(pyautogui.hotkey, *keys)
    return f"pressed {'+'.join(keys)}", False


async def _h_hold_key(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    combo = action_input.get("text", "")
    duration = float(action_input.get("duration", 1.0))
    if not isinstance(combo, str) or not combo:
        return "hold_key: 'text' must be a non-empty string", True
    keys = [_translate_key(k) for k in combo.split("+")]
    for k in keys:
        await asyncio.to_thread(pyautogui.keyDown, k)
    try:
        await asyncio.sleep(duration)
    finally:
        for k in reversed(keys):
            await asyncio.to_thread(pyautogui.keyUp, k)
    return f"held {'+'.join(keys)} for {duration}s", False


async def _h_scroll(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    coord = action_input.get("coordinate")
    if coord:
        x, y = int(coord[0]), int(coord[1])
        await asyncio.to_thread(pyautogui.moveTo, x, y)
    direction = action_input.get("scroll_direction", "down")
    amount = int(action_input.get("scroll_amount", 3))
    sign = -1 if direction == "down" else 1 if direction == "up" else 0
    if sign == 0:
        # Horizontal scroll
        h_sign = -1 if direction == "left" else 1
        await asyncio.to_thread(pyautogui.hscroll, h_sign * amount)
    else:
        await asyncio.to_thread(pyautogui.scroll, sign * amount)
    return f"scrolled {direction} by {amount}", False


async def _h_wait(action_input: dict, _browser: Browser) -> tuple[Any, bool]:
    duration = float(action_input.get("duration", 1.0))
    await asyncio.sleep(duration)
    return f"waited {duration}s", False


async def _h_cursor_position(
    _action_input: dict, _browser: Browser
) -> tuple[Any, bool]:
    pos = await asyncio.to_thread(pyautogui.position)
    return f"X={pos[0]},Y={pos[1]}", False


_ACTION_HANDLERS: dict[str, Callable[[dict, Browser], Awaitable[tuple[Any, bool]]]] = {
    "screenshot": _h_screenshot,
    "mouse_move": _h_mouse_move,
    "left_click": _h_left_click,
    "click": _h_left_click,  # legacy alias
    "right_click": _h_right_click,
    "middle_click": _h_middle_click,
    "double_click": _h_double_click,
    "triple_click": _h_triple_click,
    "left_click_drag": _h_left_click_drag,
    "left_mouse_down": _h_left_mouse_down,
    "left_mouse_up": _h_left_mouse_up,
    "type": _h_type,
    "key": _h_key,
    "hold_key": _h_hold_key,
    "scroll": _h_scroll,
    "wait": _h_wait,
    "cursor_position": _h_cursor_position,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(action_input: dict, field: str) -> tuple[int, int]:
    coord = action_input.get(field)
    if not coord or len(coord) != 2:
        raise ValueError(f"missing/invalid {field}: {coord!r}")
    return int(coord[0]), int(coord[1])


def _translate_key(key: str) -> str:
    key = key.strip()
    return _CLAUDE_TO_PYAUTOGUI_KEY.get(key, key.lower())


def _image_block(b64_png: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64_png,
        },
    }


async def _capture_image_block(browser: Browser) -> list[dict]:
    # Settle delay before the post-action screenshot. Tuned for RDP, where
    # the worker → freerdp → host → redraw → back round-trip can easily take
    # several hundred ms; sub-second settles risk capturing the pre-action
    # frame and confusing the model into "I must have missed, try again".
    await asyncio.sleep(2.0)
    screenshot = await browser.get_screenshot()
    if not screenshot:
        raise RuntimeError("get_screenshot returned None")
    return [_image_block(screenshot)]


def _block_type(block: Any) -> str | None:
    return getattr(block, "type", None) or (
        block.get("type") if isinstance(block, dict) else None
    )


def _block_input(block: Any) -> dict | None:
    return getattr(block, "input", None) or (
        block.get("input") if isinstance(block, dict) else None
    )


def _block_id(block: Any) -> str | None:
    return getattr(block, "id", None) or (
        block.get("id") if isinstance(block, dict) else None
    )


def _join_text_blocks(blocks: list) -> str | None:
    parts: list[str] = []
    for b in blocks:
        if _block_type(b) == "text":
            text = getattr(b, "text", None) or (
                b.get("text") if isinstance(b, dict) else None
            )
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else None


def _serialize_blocks(blocks: list) -> list[dict]:
    """Convert SDK content blocks to plain JSON-serializable dicts suitable
    for both logging and re-sending in the next ``messages`` payload."""
    out: list[dict] = []
    for b in blocks:
        if isinstance(b, dict):
            out.append(b)
            continue
        # Anthropic SDK pydantic models
        if hasattr(b, "model_dump"):
            out.append(b.model_dump(exclude_none=True))
        else:
            out.append({"type": getattr(b, "type", "unknown")})
    return out


def _write_json(path, payload) -> None:
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"computer_use: failed to write {path}: {e}")


def _save_screenshot_b64(b64_png: str, path: Path) -> None:
    """Persist a base64-encoded PNG to disk for offline debugging."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_png))
    except Exception as e:
        logger.warning(f"computer_use: failed to write screenshot {path}: {e}")
