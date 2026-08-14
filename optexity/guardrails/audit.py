import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GuardrailAuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str
    decision: Literal["allow", "deny", "audit"]
    event_type: str
    code: str
    action: str | None = None
    source: str | None = None
    current_url: str | None = None
    target_url: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GuardrailAuditLogger:
    """Append-only, secret-safe JSONL audit trail for policy decisions."""

    def __init__(self, task_id: str, path: Path | None = None):
        self.task_id = task_id
        self.path = path
        self.events: list[GuardrailAuditEvent] = []
        self._lock = Lock()

    def emit(self, **kwargs) -> GuardrailAuditEvent:
        event = GuardrailAuditEvent(task_id=self.task_id, **kwargs)
        with self._lock:
            self.events.append(event)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as audit_file:
                    audit_file.write(event.model_dump_json() + "\n")
        log_fn = logger.warning if event.decision == "deny" else logger.info
        log_fn(
            "guardrail decision=%s code=%s action=%s source=%s",
            event.decision,
            event.code,
            event.action,
            event.source,
        )
        return event
