import base64
import logging
from pathlib import Path
from typing import List, Optional

import aiofiles
from pydantic import create_model

logger = logging.getLogger(__name__)


def build_model(schema: dict, model_name="AutoModel"):
    fields = {}
    for key, value in schema.items():
        if isinstance(value, str):  # primitive type
            py_type = eval(value)  # e.g., "str" -> str
            fields[key] = (Optional[py_type], None)
        elif isinstance(value, dict):  # nested object
            sub_model = build_model(value, model_name=f"{model_name}_{key}")
            fields[key] = (Optional[sub_model], None)
        elif isinstance(value, list):  # list of objects or primitives
            if len(value) > 0 and isinstance(value[0], dict):
                sub_model = build_model(value[0], model_name=f"{model_name}_{key}")
                fields[key] = (Optional[List[sub_model]], None)
            else:  # list of primitives
                py_type = eval(value[0])
                fields[key] = (Optional[List[py_type]], None)
    return create_model(model_name, **fields)


def save_screenshot(screenshot: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(screenshot))


async def save_and_clear_downloaded_files(content: bytes | str, filename: Path):
    if isinstance(content, bytes):
        async with aiofiles.open(filename, "wb") as f:
            await f.write(content)
    elif isinstance(content, str):
        async with aiofiles.open(filename, "w") as f:
            await f.write(content)
    else:
        logger.error(f"Unsupported content type: {type(content)}")
