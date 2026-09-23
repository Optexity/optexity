import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("optexity")
except PackageNotFoundError:
    __version__ = "0.0.0"

logging.basicConfig(
    level=logging.WARNING,  # Default level for root logger
    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("/tmp/optexity.log")),
    ],
)
current_module = __name__.split(".")[0]  # top-level module/package
logging.getLogger(current_module).setLevel(logging.DEBUG)
# Portal modules log under optexity_private. The root logger stays at WARNING,
# so without this their info/debug lines are dropped before any task-log
# handler can write them.
logging.getLogger("optexity_private").setLevel(logging.DEBUG)
