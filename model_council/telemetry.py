"""Privacy-scoped SQLite telemetry for Model Council outcomes.

The schema intentionally stores task features and outcome metadata only. It does
not have prompt, output, credential, or tool-transcript columns.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path


class OutcomeKind(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class FeedbackKind(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")
_SAFE_CODE = re.compile(r"^[a-z0-9_.:-]{1,64}$")
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OutcomeEvent:
    event_id: str
    occurred_at: str
    task_kind: str
    complexity: int
    risk: int
    plan_id: str
    role: str
    provider: str
    model: str
    family: str
    outcome: OutcomeKind
    evaluator_score: float | None
    latency_ms: int
    execution_calls: int
    total_tokens: int
    failure_code: str | None
    feedback: FeedbackKind
    policy_version: str

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "task_kind",
            "plan_id",
            "role",
            "provider",
            "model",
            "family",
            "policy_version",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value or not _SAFE_LABEL.fullmatch(value):
                raise ValueError(f"{name} contains unsafe characters")
        if not _SAFE_ID.fullmatch(str(self.event_id)):
            raise ValueError("event_id contains unsafe characters")
        try:
            timestamp = datetime.fromisoformat(self.occurred_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("occurred_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        if not 1 <= int(self.complexity) <= 5:
            raise ValueError("complexity must be between 1 and 5")
        if not 1 <= int(self.risk) <= 5:
            raise ValueError("risk must be between 1 and 5")
        if self.evaluator_score is not None and not 0.0 <= float(self.evaluator_score) <= 1.0:
            raise ValueError("evaluator_score must be between 0 and 1")
        if int(self.latency_ms) < 0 or int(self.execution_calls) < 0 or int(self.total_tokens) < 0:
            raise ValueError("latency, calls, and tokens must not be negative")
        if self.failure_code is not None and not _SAFE_CODE.fullmatch(self.failure_code):
            raise ValueError("failure_code contains unsafe characters")
        if not isinstance(self.outcome, OutcomeKind):
            raise ValueError("outcome must be an OutcomeKind")
        if not isinstance(self.feedback, FeedbackKind):
            raise ValueError("feedback must be a FeedbackKind")


@dataclass(frozen=True)
class PerformanceSummary:
    task_kind: str
    provider: str
    model: str
    family: str
    sample_count: int
    successes: int
    failures: int
    mean_score: float | None
    mean_latency_ms: float | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcome_events (
    event_id TEXT PRIMARY KEY,
    occurred_at TEXT NOT NULL,
    task_kind TEXT NOT NULL,
    complexity INTEGER NOT NULL,
    risk INTEGER NOT NULL,
    plan_id TEXT NOT NULL,
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    family TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown')),
    evaluator_score REAL,
    latency_ms INTEGER NOT NULL,
    execution_calls INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    failure_code TEXT,
    feedback TEXT NOT NULL CHECK (feedback IN ('positive', 'negative', 'none')),
    policy_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcome_task_model
    ON outcome_events(task_kind, provider, model);
"""


class TelemetryStore:
    """A local, prompt-free SQLite store for outcome metadata and aggregates."""

    def __init__(self, path: Path, *, retention_days: int = 90) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(1, int(retention_days))
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            elif row[0] != str(_SCHEMA_VERSION):
                raise RuntimeError(f"unsupported telemetry schema version: {row[0]}")
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _utc_timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc).isoformat()

    def record(self, event: OutcomeEvent, *, now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        cutoff = current - timedelta(days=self.retention_days)
        occurred_at = self._utc_timestamp(datetime.fromisoformat(event.occurred_at))
        connection = self._connect()
        try:
            connection.execute(
                "DELETE FROM outcome_events WHERE occurred_at < ?",
                (self._utc_timestamp(cutoff),),
            )
            if occurred_at < self._utc_timestamp(cutoff):
                connection.commit()
                return
            try:
                connection.execute(
                    """INSERT INTO outcome_events (
                        event_id, occurred_at, task_kind, complexity, risk, plan_id,
                        role, provider, model, family, outcome, evaluator_score,
                        latency_ms, execution_calls, total_tokens, failure_code,
                        feedback, policy_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.event_id,
                        occurred_at,
                        event.task_kind,
                        event.complexity,
                        event.risk,
                        event.plan_id,
                        event.role,
                        event.provider,
                        event.model,
                        event.family,
                        event.outcome.value,
                        event.evaluator_score,
                        event.latency_ms,
                        event.execution_calls,
                        event.total_tokens,
                        event.failure_code,
                        event.feedback.value,
                        event.policy_version,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate telemetry event: {event.event_id}") from exc
            connection.commit()
        finally:
            connection.close()

    def count_events(self) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM outcome_events").fetchone()[0])
        finally:
            connection.close()

    def summarize(self, *, task_kind: str | None = None) -> tuple[PerformanceSummary, ...]:
        connection = self._connect()
        try:
            if task_kind is None:
                rows = connection.execute(
                    """SELECT task_kind, provider, model, family, COUNT(*),
                              SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END),
                              AVG(evaluator_score), AVG(latency_ms)
                       FROM outcome_events
                      GROUP BY task_kind, provider, model, family
                      ORDER BY task_kind, provider, model"""
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT task_kind, provider, model, family, COUNT(*),
                              SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END),
                              AVG(evaluator_score), AVG(latency_ms)
                       FROM outcome_events
                      WHERE task_kind = ?
                      GROUP BY task_kind, provider, model, family
                      ORDER BY provider, model""",
                    (task_kind,),
                ).fetchall()
            return tuple(PerformanceSummary(*row) for row in rows)
        finally:
            connection.close()

    def integrity_check(self) -> bool:
        connection = self._connect()
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            return bool(result and result[0] == "ok")
        finally:
            connection.close()

    def schema_version(self) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                raise RuntimeError("telemetry schema version is missing")
            return int(row[0])
        finally:
            connection.close()
