"""Privacy-scoped SQLite telemetry for Model Council outcomes.

The schema intentionally stores task features and outcome metadata only. It does
not have prompt, output, credential, or tool-transcript columns.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

from .analysis import TaskProfile
from .inventory import ModelSpec


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
_SCHEMA_VERSION = 2


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
    total_tokens: int | None
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
        if (
            int(self.latency_ms) < 0
            or int(self.execution_calls) < 0
            or (self.total_tokens is not None and int(self.total_tokens) < 0)
        ):
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
    unknown_outcomes: int
    positive_feedback: int
    negative_feedback: int
    mean_score: float | None
    mean_latency_ms: float | None
    mean_execution_calls: float | None
    mean_total_tokens: float | None


@dataclass(frozen=True)
class RunPerformanceSummary:
    task_kind: str
    plan_id: str
    sample_count: int
    successes: int
    failures: int
    unknown_outcomes: int
    positive_feedback: int
    negative_feedback: int
    mean_score: float | None
    mean_latency_ms: float | None
    mean_execution_calls: float | None
    mean_total_tokens: float | None


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
    total_tokens INTEGER,
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
            elif row[0] == "1":
                self._migrate_v1_to_v2(connection)
            elif row[0] != str(_SCHEMA_VERSION):
                raise RuntimeError(f"unsupported telemetry schema version: {row[0]}")
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        """Make token counts nullable while preserving all v1 outcome rows."""
        try:
            connection.execute("DROP INDEX IF EXISTS idx_outcome_task_model")
            connection.execute("ALTER TABLE outcome_events RENAME TO outcome_events_v1")
            connection.execute(
                """CREATE TABLE outcome_events (
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
                    total_tokens INTEGER,
                    failure_code TEXT,
                    feedback TEXT NOT NULL CHECK (feedback IN ('positive', 'negative', 'none')),
                    policy_version TEXT NOT NULL
                )"""
            )
            connection.execute(
                """INSERT INTO outcome_events (
                    event_id, occurred_at, task_kind, complexity, risk, plan_id,
                    role, provider, model, family, outcome, evaluator_score,
                    latency_ms, execution_calls, total_tokens, failure_code,
                    feedback, policy_version
                )
                SELECT event_id, occurred_at, task_kind, complexity, risk, plan_id,
                       role, provider, model, family, outcome, evaluator_score,
                       latency_ms, execution_calls, total_tokens, failure_code,
                       feedback, policy_version
                  FROM outcome_events_v1"""
            )
            connection.execute("DROP TABLE outcome_events_v1")
            connection.execute(
                """CREATE INDEX idx_outcome_task_model
                       ON outcome_events(task_kind, provider, model)"""
            )
            connection.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(_SCHEMA_VERSION),),
            )
        except Exception:
            connection.rollback()
            raise

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

    def record_run_outcome(
        self,
        *,
        event_id: str,
        task_profile: TaskProfile,
        plan_id: str,
        outcome: OutcomeKind,
        evaluator_score: float | None,
        feedback: FeedbackKind,
        latency_ms: int,
        execution_calls: int,
        failure_code: str | None = None,
        total_tokens: int | None = None,
        policy_version: str = "telemetry-v2",
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(timezone.utc)
        self.record(
            OutcomeEvent(
                event_id=event_id,
                occurred_at=current.isoformat(),
                task_kind=task_profile.kind,
                complexity=task_profile.complexity,
                risk=task_profile.risk,
                plan_id=plan_id,
                role="run",
                provider="council",
                model="council",
                family="council",
                outcome=outcome,
                evaluator_score=evaluator_score,
                latency_ms=latency_ms,
                execution_calls=execution_calls,
                total_tokens=total_tokens,
                failure_code=failure_code,
                feedback=feedback,
                policy_version=policy_version,
            ),
            now=current,
        )

    def record_outcome_for_run(
        self,
        *,
        run_id: str,
        outcome: OutcomeKind,
        evaluator_score: float | None,
        feedback: FeedbackKind,
        failure_code: str | None = None,
        total_tokens: int | None = None,
        policy_version: str = "telemetry-v2",
        now: datetime | None = None,
    ) -> str:
        if not _SAFE_ID.fullmatch(run_id) or len(run_id) > 56:
            raise ValueError("run_id must be a safe identifier of at most 56 characters")
        event_id = f"{run_id}:outcome"
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT task_kind, complexity, risk, plan_id,
                          COUNT(*), SUM(latency_ms), SUM(execution_calls)
                     FROM outcome_events
                    WHERE substr(event_id, 1, ?) = ? AND role != 'run'
                    GROUP BY task_kind, complexity, risk, plan_id""",
                (len(run_id) + 1, f"{run_id}:"),
            ).fetchall()
            if not rows:
                raise ValueError(f"no telemetry calls found for run_id: {run_id}")
            if len(rows) != 1:
                raise ValueError(f"telemetry run has inconsistent metadata: {run_id}")
            existing = connection.execute(
                "SELECT 1 FROM outcome_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if existing:
                raise ValueError(f"run outcome already recorded: {run_id}")
            task_kind, complexity, risk, plan_id, calls, latency, _ = rows[0]
        finally:
            connection.close()
        profile = TaskProfile(str(task_kind), int(complexity), int(risk), False, False, False)
        self.record_run_outcome(
            event_id=event_id,
            task_profile=profile,
            plan_id=str(plan_id),
            outcome=outcome,
            evaluator_score=evaluator_score,
            feedback=feedback,
            latency_ms=int(latency or 0),
            execution_calls=int(calls or 0),
            failure_code=failure_code,
            total_tokens=total_tokens,
            policy_version=policy_version,
            now=now,
        )
        return event_id

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
                              SUM(CASE WHEN outcome = 'unknown' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END),
                              AVG(evaluator_score), AVG(latency_ms),
                              AVG(execution_calls), AVG(total_tokens)
                       FROM outcome_events
                      GROUP BY task_kind, provider, model, family
                      ORDER BY task_kind, provider, model"""
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT task_kind, provider, model, family, COUNT(*),
                              SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN outcome = 'unknown' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END),
                              SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END),
                              AVG(evaluator_score), AVG(latency_ms),
                              AVG(execution_calls), AVG(total_tokens)
                       FROM outcome_events
                      WHERE task_kind = ?
                      GROUP BY task_kind, provider, model, family
                      ORDER BY provider, model""",
                    (task_kind,),
                ).fetchall()
            return tuple(PerformanceSummary(*row) for row in rows)
        finally:
            connection.close()

    def summarize_runs(
        self, *, task_kind: str | None = None
    ) -> tuple[RunPerformanceSummary, ...]:
        connection = self._connect()
        try:
            where = "WHERE role = 'run'"
            parameters: tuple[str, ...] = ()
            if task_kind is not None:
                where += " AND task_kind = ?"
                parameters = (task_kind,)
            rows = connection.execute(
                f"""SELECT task_kind, plan_id, COUNT(*),
                           SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN outcome = 'unknown' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END),
                           AVG(evaluator_score), AVG(latency_ms),
                           AVG(execution_calls), AVG(total_tokens)
                      FROM outcome_events
                      {where}
                     GROUP BY task_kind, plan_id
                     ORDER BY task_kind, plan_id""",
                parameters,
            ).fetchall()
            return tuple(RunPerformanceSummary(*row) for row in rows)
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


class TelemetryInvoker:
    """Record per-call metadata without persisting prompts or outputs."""

    def __init__(
        self,
        *,
        invoke: Callable[[ModelSpec, str, str, str], str],
        store: TelemetryStore,
        task_profile: TaskProfile,
        plan_id: str,
        run_id: str,
        policy_version: str = "telemetry-v1",
    ) -> None:
        if not _SAFE_ID.fullmatch(run_id) or len(run_id) > 56:
            raise ValueError("run_id must be a safe identifier of at most 56 characters")
        if not _SAFE_LABEL.fullmatch(plan_id):
            raise ValueError("plan_id contains unsafe characters")
        self.invoke = invoke
        self.store = store
        self.task_profile = task_profile
        self.plan_id = plan_id
        self.run_id = run_id
        self.policy_version = policy_version
        self._counter = 0
        self._counter_lock = threading.Lock()

    def _next_event_id(self, role: str) -> str:
        with self._counter_lock:
            self._counter += 1
            ordinal = self._counter
        safe_role = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(role))[:64] or "unknown"
        return f"{self.run_id}:{ordinal}:{safe_role}"

    @staticmethod
    def _failure_code(exc: Exception) -> str:
        text = str(exc or "").casefold()
        if "timeout" in text or isinstance(exc, TimeoutError):
            return "timeout"
        if "429" in text or "rate limit" in text:
            return "rate_limited"
        return "invocation_failure"

    def _record(
        self,
        *,
        event_id: str,
        model: ModelSpec,
        role: str,
        outcome: OutcomeKind,
        latency_ms: int,
        failure_code: str | None,
    ) -> None:
        try:
            self.store.record(
                OutcomeEvent(
                    event_id=event_id,
                    occurred_at=datetime.now(timezone.utc).isoformat(),
                    task_kind=self.task_profile.kind,
                    complexity=self.task_profile.complexity,
                    risk=self.task_profile.risk,
                    plan_id=self.plan_id,
                    role=role,
                    provider=model.provider,
                    model=model.model,
                    family=model.family,
                    outcome=outcome,
                    evaluator_score=None,
                    latency_ms=max(0, int(latency_ms)),
                    execution_calls=1,
                    total_tokens=None,
                    failure_code=failure_code,
                    feedback=FeedbackKind.NONE,
                    policy_version=self.policy_version,
                )
            )
        except Exception:
            # Telemetry is best-effort and must not change the model-call path.
            return

    def __call__(
        self, model: ModelSpec, prompt: str, role: str, reasoning_effort: str
    ) -> str:
        event_id = self._next_event_id(role)
        started = time.perf_counter()
        try:
            result = self.invoke(model, prompt, role, reasoning_effort)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self._record(
                event_id=event_id,
                model=model,
                role=role,
                outcome=OutcomeKind.FAILURE,
                latency_ms=elapsed_ms,
                failure_code=self._failure_code(exc),
            )
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._record(
            event_id=event_id,
            model=model,
            role=role,
            outcome=OutcomeKind.SUCCESS,
            latency_ms=elapsed_ms,
            failure_code=None,
        )
        return result
