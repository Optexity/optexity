import logging

from optexity.inference.infra.browser import Browser
from optexity.schema.actions.misc_action import PythonScriptAction
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)


async def run_python_script_action(
    python_script_action: PythonScriptAction, memory: Memory, browser: Browser
):
    local_vars = {}
    exec(python_script_action.execution_code, {}, local_vars)

    # Get the function
    code_fn = local_vars["code_fn"]

    page = await browser.get_current_page()
    await code_fn(page)
