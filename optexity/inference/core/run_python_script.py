import logging

from optexity.inference.core.script_context import ScriptContext, call_script_fn
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.misc_action import PythonScriptAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def run_python_script_action(
    python_script_action: PythonScriptAction,
    memory: Memory,
    browser: Browser,
    task: Task,
):
    local_vars = {}
    exec(python_script_action.execution_code, {}, local_vars)

    # Get the function
    code_fn = local_vars["code_fn"]

    page = await browser.get_current_page()
    ctx = ScriptContext(task=task, memory=memory, browser=browser)
    await call_script_fn(code_fn, (page,), ctx)
