"""Extension point for node handlers that live outside this package.

The public SDK owns the ``private_node`` schema and this registry; closed-source
distributions ship a separate package that registers handlers against it at
import time, advertised through the ``optexity.plugins`` entry-point group.
Nothing here imports a plugin by name, so the public SDK builds and runs with no
plugin installed — a ``private_node`` naming an absent handler then fails at that
node with ``HandlerNotRegistered`` while the rest of the automation proceeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from optexity.inference.core.script_context import ScriptContext

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "optexity.plugins"


class HandlerNotRegistered(Exception):
    def __init__(self, handler: str, available: list[str]):
        self.handler = handler
        super().__init__(
            f"No handler registered for {handler!r}. Registered handlers: "
            f"{available or 'none — no plugin package is installed'}"
        )


@dataclass(frozen=True)
class HandlerSpec:
    """One callable addressable from a ``private_node``.

    ``run`` receives the validated inputs and the run's ``ScriptContext``. When
    ``inputs_model`` is set the node's raw ``inputs`` dict is validated through it
    first, so a handler never has to defend against a malformed payload.
    """

    name: str
    run: Callable[[Any, "ScriptContext"], Awaitable[Any]]
    inputs_model: type[BaseModel] | None = None


class HandlerRegistry:
    _handlers: ClassVar[dict[str, HandlerSpec]] = {}

    @classmethod
    def register(cls, spec: HandlerSpec) -> None:
        if spec.name in cls._handlers:
            raise ValueError(
                f"handler {spec.name!r} is already registered; two plugins claim "
                f"the same name"
            )
        cls._handlers[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> HandlerSpec:
        try:
            return cls._handlers[name]
        except KeyError:
            raise HandlerNotRegistered(name, cls.names()) from None

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._handlers)


_plugins_loaded = False


def load_plugins() -> list[str]:
    """Import and register every installed plugin package. Idempotent.

    A plugin that raises on load is logged and skipped rather than aborting the
    run: only the nodes that need its handlers fail, and they fail with
    ``HandlerNotRegistered`` naming what is actually available.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return HandlerRegistry.names()
    _plugins_loaded = True

    for entry_point in entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        try:
            entry_point.load()()
        except Exception as e:
            logger.error(f"Failed to load optexity plugin {entry_point.name!r}: {e}")

    registered = HandlerRegistry.names()
    logger.info(f"Loaded private node handlers: {registered or 'none'}")
    return registered
