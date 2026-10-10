"""Memory layer: remember what browser-use did, and replay it without the LLM.

Public pieces:
    ActionCache, ActionNote   - the shape of the raw notes file (schema.py)
    compile_cache             - notes in, Optexity workflow out (compiler.py)

The recorder (recorder.py) is imported on demand because it needs the browser
side of Optexity; importing this package stays light.
"""

from optexity.memory.compiler import CompileResult, compile_cache, compile_file
from optexity.memory.schema import ActionCache, ActionNote

__all__ = [
    "ActionCache",
    "ActionNote",
    "CompileResult",
    "compile_cache",
    "compile_file",
]
