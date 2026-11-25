"""Local support tickets; synchronous SQLite access for asyncio.to_thread callers."""
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class TicketStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS tickets (
                        ticket_id TEXT PRIMARY KEY,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        user_id TEXT NOT NULL,
                        conv_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        submission_text TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
        finally:
            connection.close()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _ticket(row):
        ticket = dict(row)
        ticket.pop("idempotency_key", None)
        ticket["snapshot"] = json.loads(ticket.pop("snapshot_json"))
        return ticket

    def create_ticket(self, user_id, conv_id, case_id, summary, snapshot, request_id, submission_text):
        for name, value in (("user_id", user_id), ("conv_id", conv_id), ("case_id", case_id),
                            ("summary", summary), ("request_id", request_id), ("submission_text", submission_text)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be an object")
        identity = json.dumps([user_id, conv_id, case_id], ensure_ascii=False, separators=(",", ":"))
        idempotency_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        snapshot_json = json.dumps(snapshot, ensure_ascii=False)
        ticket_id = "NEXO-" + uuid.uuid4().hex.upper()
        created_at = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute("""
                    INSERT INTO tickets (
                        ticket_id, idempotency_key, user_id, conv_id, case_id, status,
                        summary, snapshot_json, request_id, submission_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(idempotency_key) DO NOTHING
                """, (ticket_id, idempotency_key, user_id, conv_id, case_id, "pending",
                      summary, snapshot_json, request_id, submission_text, created_at))
                created = cursor.rowcount == 1
                row = connection.execute(
                    "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,),
                ).fetchone()
                ticket = self._ticket(row)
            return {"success": True, "created": created, "ticket": ticket}
        finally:
            connection.close()

    def get_ticket(self, ticket_id, user_id, conv_id):
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ? AND user_id = ? AND conv_id = ?",
                (ticket_id, user_id, conv_id),
            ).fetchone()
            return {"success": True, "found": row is not None,
                    "ticket": self._ticket(row) if row is not None else None}
        finally:
            connection.close()
