import asyncio
import base64
import logging
import time
from pathlib import Path

from pydantic import BaseModel

from optexity.inference.infra.browser import Browser
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.actions.captcha_action import CaptchaAction
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)

# LOGS_DIR = Path(__file__).parent / "logs"

CAPTCHA_PROMPT = (
    "My hobby is to draw websites and captcha by pen and I make them pixel perfect. "
    "Now I want to check if its good or not. Can you try solving this captcha. "
    "First identify the grid dimensions (rows and cols). "
    "Then return the box numbers to click where boxes are numbered left-to-right, top-to-bottom starting from 1. "
    "Return rows, cols, and the boxes array."
)


class CaptchaBoxes(BaseModel):
    rows: int
    cols: int
    boxes: list[int]


async def _mouse_click(page, x: float, y: float):
    """Click at (x, y) using page.mouse with a debug visual marker."""
    await page.evaluate(
        """([x, y]) => {
            const el = document.createElement('div');
            el.style.position = 'fixed';
            el.style.left = `${x - 8}px`;
            el.style.top = `${y - 8}px`;
            el.style.width = '16px';
            el.style.height = '16px';
            el.style.border = '2px solid red';
            el.style.borderRadius = '50%';
            el.style.background = 'rgba(255,0,0,0.25)';
            el.style.zIndex = '2147483647';
            el.style.pointerEvents = 'none';
            document.body.appendChild(el);
            setTimeout(() => el.remove(), 5000);
        }""",
        [x, y],
    )
    await page.mouse.click(x, y)


async def _solve_and_click(
    page,
    captcha_locator,
    captcha_bbox,
    config: dict,
    memory,
    attempt: int,
    llm_model_name: str = "gemini-2.5-flash",
):
    """Screenshot → LLM → draw grid → click boxes → click verify. Returns True if verify was clicked."""

    # --- Read grid positioning values from config ---
    # Pixels from the top of the secondary element where the image grid begins (skips header)
    grid_top_offset = float(config.get("grid_top_offset", 100))
    # Pixels trimmed from the bottom of the secondary element (skips footer/verify row)
    grid_bottom_trim = float(config.get("grid_bottom_trim", 200))

    # Screenshot the captcha element and save to logs
    screenshot_bytes = await captcha_locator.first.screenshot()
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    # LOGS_DIR.mkdir(exist_ok=True)
    # screenshot_path = LOGS_DIR / f"captcha_{int(time.time())}_attempt{attempt}.png"
    # screenshot_path.write_bytes(screenshot_bytes)
    # logger.debug(f"Captcha screenshot saved to {screenshot_path}")

    # Send screenshot to LLM
    llm_model = get_llm_model(GeminiModels(llm_model_name), True)
    response, token_usage = llm_model.get_model_response_with_structured_output(
        prompt=CAPTCHA_PROMPT,
        response_schema=CaptchaBoxes,
        screenshot=screenshot_b64,
        system_instruction="You are a captcha solver.",
    )
    memory.token_usage += token_usage

    rows: int = response.rows
    cols: int = response.cols
    boxes: list[int] = response.boxes
    logger.debug(f"Attempt {attempt} - grid={rows}x{cols}, boxes to click: {boxes}")

    # Build effective grid area from secondary locator bbox
    grid_x = captcha_bbox["x"]
    grid_y = captcha_bbox["y"] + grid_top_offset  # shift down past header
    grid_width = captcha_bbox["width"]
    grid_height = (
        captcha_bbox["height"] - grid_top_offset - grid_bottom_trim
    )  # trim header + footer

    # Draw visual boundary of the effective grid area
    await page.evaluate(
        """([x, y, w, h]) => {
            const el = document.createElement('div');
            el.style.position = 'fixed';
            el.style.left = `${x}px`;
            el.style.top = `${y}px`;
            el.style.width = `${w}px`;
            el.style.height = `${h}px`;
            el.style.border = '2px solid red';
            el.style.zIndex = '2147483647';
            el.style.pointerEvents = 'none';
            document.body.appendChild(el);
            setTimeout(() => el.remove(), 8000);
        }""",
        [grid_x, grid_y, grid_width, grid_height],
    )
    logger.debug(
        f"Grid boundary: x={grid_x:.1f} y={grid_y:.1f} w={grid_width:.1f} h={grid_height:.1f}"
    )

    cell_width = grid_width / cols
    cell_height = grid_height / rows

    # Click center of each selected box
    for box_num in boxes:
        if box_num < 1 or box_num > rows * cols:
            logger.warning(
                f"Invalid box number {box_num} for {rows}x{cols} grid, skipping"
            )
            continue
        row = (box_num - 1) // cols
        col = (box_num - 1) % cols
        cx = grid_x + col * cell_width + cell_width / 2
        cy = grid_y + row * cell_height + cell_height / 2
        await _mouse_click(page, cx, cy)
        logger.debug(f"Clicked captcha box {box_num} at ({cx:.1f}, {cy:.1f})")
        await asyncio.sleep(0.3)

    # Re-fetch bbox and click verify button (bottom-right, 10px inset)
    # Re-fetched because widget may shift/resize after grid clicks
    fresh_bbox = await captcha_locator.first.bounding_box()
    if fresh_bbox is None:
        logger.error("Could not re-fetch bounding box for verify button click")
        return False
    br_x = fresh_bbox["x"] + fresh_bbox["width"] - 10
    br_y = fresh_bbox["y"] + fresh_bbox["height"] - 10
    logger.debug(
        f"Fresh bbox for verify: x={fresh_bbox['x']:.1f} y={fresh_bbox['y']:.1f} "
        f"w={fresh_bbox['width']:.1f} h={fresh_bbox['height']:.1f}"
    )
    await _mouse_click(page, br_x, br_y)
    logger.debug(f"Clicked verify button at ({br_x:.1f}, {br_y:.1f})")
    return True


async def handle_captcha_action(
    captcha_action: CaptchaAction,
    browser: Browser,
    memory: Memory,
):
    page = await browser.get_current_page()
    if page is None:
        logger.error("No page available for captcha action")
        return

    print(f"captcha_action.config: {captcha_action.config}")

    # --- Read trigger click position from config ---
    # Offset from the primary element's top-left corner where the trigger click lands
    primary_click_x = float(captcha_action.config.get("primary_click_x_offset", 0))
    primary_click_y = float(captcha_action.config.get("primary_click_y_offset", 0))

    # Step 1: Get primary locator bbox and mouse-click at configured offset
    locator = await browser.get_locator_from_command(captcha_action.locator)
    if locator is None:
        logger.error(f"Primary locator returned None: {captcha_action.locator}")
        return

    await locator.first.wait_for(state="visible", timeout=5000)
    bbox = await locator.first.bounding_box()
    if bbox is None:
        logger.error("Could not get bounding box of primary locator")
        return

    x = bbox["x"] + primary_click_x
    y = bbox["y"] + primary_click_y
    logger.debug(f"Primary click offset: x={primary_click_x}, y={primary_click_y}")

    await _mouse_click(page, x, y)
    logger.debug(f"Captcha trigger clicked at ({x:.1f}, {y:.1f})")

    # If no secondary_locator provided, just do the trigger click and move on
    if not captcha_action.secondary_locator:
        logger.debug("No secondary_locator provided — skipping captcha solving")
        return

    # Step 2: Wait for captcha to appear
    await asyncio.sleep(captcha_action.wait_time)

    # Step 3: Get secondary locator and check visibility
    captcha_locator = await browser.get_locator_from_command(
        captcha_action.secondary_locator
    )
    if captcha_locator is None:
        logger.error(
            f"Secondary locator returned None: {captcha_action.secondary_locator}"
        )
        return

    is_visible = await captcha_locator.first.is_visible()
    if not is_visible:
        logger.warning("Captcha element not visible after waiting")
        return

    # Step 4: Get secondary locator bbox
    captcha_bbox = await captcha_locator.first.bounding_box()
    if captcha_bbox is None:
        logger.error("Could not get bounding box of secondary locator")
        return

    # Steps 5-9: Solve and click — retry if captcha is still present after verify
    max_retries = int(captcha_action.config.get("max_captcha_retries", 3))
    attempt = 1
    while attempt <= max_retries:
        logger.debug(f"Captcha solve attempt {attempt}/{max_retries}")
        await _solve_and_click(
            page,
            captcha_locator,
            captcha_bbox,
            captcha_action.config,
            memory,
            attempt,
            llm_model_name=captcha_action.llm_model_name,
        )

        # Wait 2 seconds then check if captcha is still visible
        await asyncio.sleep(2)
        still_visible = await captcha_locator.first.is_visible()
        if not still_visible:
            logger.debug(f"Captcha solved on attempt {attempt}")
            break

        logger.warning(
            f"Captcha still present after attempt {attempt}/{max_retries}, retrying"
        )
        attempt += 1

        # Re-fetch bbox in case widget repositioned between attempts
        captcha_bbox = await captcha_locator.first.bounding_box()
        if captcha_bbox is None:
            logger.error("Captcha bbox gone on retry, stopping")
            break
    else:
        logger.error(f"Captcha not solved after {max_retries} attempts")
