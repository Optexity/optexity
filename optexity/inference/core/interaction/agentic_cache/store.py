"""SQLite-backed store for synthesized deterministic-node caches, keyed by
automation_key = sha256(url + agentic_task.task text). One row per capture ("block"),
never updated in place -- history is kept via the version column.
"""

import contextlib
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from optexity.inference.core.interaction.agentic_cache.models import CachedBlock
from optexity.schema.automation import ActionNode, Automation
from optexity.utils.settings import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cached_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    automation_key TEXT NOT NULL,
    nodes_json TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_verified_at TEXT,
    source_run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_cached_blocks_key ON cached_blocks(automation_key);
"""


def compute_automation_key(url: str, task_text: str) -> str:
    return hashlib.sha256(f"{url}\n{task_text}".encode("utf-8")).hexdigest()


def _row_to_block(row: sqlite3.Row) -> CachedBlock:
    nodes = [ActionNode.model_validate(n) for n in json.loads(row["nodes_json"])]
    return CachedBlock(
        id=row["id"],
        automation_key=row["automation_key"],
        nodes=nodes,
        verified=bool(row["verified"]),
        version=row["version"],
        created_at=row["created_at"],
        last_verified_at=row["last_verified_at"],
        source_run_id=row["source_run_id"],
    )


class CachedBlockStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else Path(settings.AGENTIC_CACHE_DB_PATH)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save(
        self,
        automation_key: str,
        nodes: list[ActionNode],
        source_run_id: str | None = None,
        verified: bool = False,
    ) -> int:
        nodes_json = json.dumps([n.model_dump(mode="json") for n in nodes])
        with self._connect() as conn:
            (next_version,) = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM cached_blocks WHERE automation_key = ?",
                (automation_key,),
            ).fetchone()
            cursor = conn.execute(
                """
                INSERT INTO cached_blocks
                    (automation_key, nodes_json, verified, version, created_at, source_run_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    automation_key,
                    nodes_json,
                    int(verified),
                    next_version,
                    datetime.now(timezone.utc).isoformat(),
                    source_run_id,
                ),
            )
            return cursor.lastrowid

    def latest(self, automation_key: str) -> CachedBlock | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cached_blocks WHERE automation_key = ? ORDER BY id DESC LIMIT 1",
                (automation_key,),
            ).fetchone()
        return _row_to_block(row) if row is not None else None

    def latest_verified(self, automation_key: str) -> CachedBlock | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM cached_blocks
                WHERE automation_key = ? AND verified = 1
                ORDER BY id DESC LIMIT 1
                """,
                (automation_key,),
            ).fetchone()
        return _row_to_block(row) if row is not None else None

    def mark_verified(self, row_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE cached_blocks SET verified = 1, last_verified_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), row_id),
            )


def export_cached_automation(
    base_automation: Automation, automation_key: str, store: CachedBlockStore
) -> Automation:
    """A copy of base_automation with the agentic_task node matching automation_key
    spliced out and replaced by the latest cached deterministic nodes. This is what
    produces the test_automation_*_cached.json baseline artifacts."""
    block = store.latest(automation_key)
    if block is None:
        raise ValueError(f"No cached block found for automation_key={automation_key}")

    new_nodes = []
    replaced = False
    for node in base_automation.nodes:
        is_match = (
            not replaced
            and node.type == "action_node"
            and node.interaction_action is not None
            and node.interaction_action.agentic_task is not None
            and compute_automation_key(
                base_automation.url, node.interaction_action.agentic_task.task
            )
            == automation_key
        )
        if is_match:
            new_nodes.extend(block.nodes)
            replaced = True
        else:
            new_nodes.append(node)

    if not replaced:
        raise ValueError(
            f"No agentic_task node in base_automation matched automation_key {automation_key!r}"
        )

    # Closes the exact gap this cache was built to observe: optexity's own
    # run_final_downloads_check only notices a download when expected_downloads is set,
    # but an agentic_task node never declares it. Since the synthesized nodes now carry
    # expect_download explicitly (see synthesize.py), roll that count up to the
    # automation level automatically instead of requiring a human to remember it.
    new_download_count = sum(
        1
        for node in block.nodes
        if node.interaction_action is not None
        and node.interaction_action.click_element is not None
        and node.interaction_action.click_element.expect_download
    )

    return base_automation.model_copy(
        update={
            "nodes": new_nodes,
            "expected_downloads": base_automation.expected_downloads + new_download_count,
        }
    )
