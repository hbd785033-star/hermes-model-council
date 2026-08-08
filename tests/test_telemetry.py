import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from model_council.analysis import TaskProfile
from model_council.inventory import ModelSpec
from model_council.telemetry import (
    FeedbackKind,
    OutcomeEvent,
    OutcomeKind,
    TelemetryInvoker,
    TelemetryStore,
)


class TelemetryStoreTests(unittest.TestCase):
    def _event(self, event_id="event-1", occurred_at="2026-08-08T12:00:00+00:00", **kwargs):
        values = {
            "event_id": event_id,
            "occurred_at": occurred_at,
            "task_kind": "security_review",
            "complexity": 4,
            "risk": 5,
            "plan_id": "quality",
            "role": "advisor",
            "provider": "provider-a",
            "model": "model-a",
            "family": "family-a",
            "outcome": OutcomeKind.SUCCESS,
            "evaluator_score": 0.9,
            "latency_ms": 1200,
            "execution_calls": 3,
            "total_tokens": 800,
            "failure_code": None,
            "feedback": FeedbackKind.POSITIVE,
            "policy_version": "router-v1",
        }
        values.update(kwargs)
        return OutcomeEvent(**values)

    def test_invoker_records_success_without_prompt_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            invoker = TelemetryInvoker(
                invoke=lambda model, prompt, role, effort: "SECRET MODEL OUTPUT",
                store=store,
                task_profile=TaskProfile("security_review", 4, 5, False, False, True),
                plan_id="quality",
                run_id="run-1",
            )
            result = invoker(
                ModelSpec("provider-a", "model-a", "family-a"),
                "SECRET USER PROMPT",
                "advisor-1",
                "high",
            )
            summary = store.summarize()

        self.assertEqual(result, "SECRET MODEL OUTPUT")
        self.assertEqual(summary[0].sample_count, 1)
        self.assertEqual(summary[0].successes, 1)
        self.assertEqual(summary[0].model, "model-a")

    def test_invoker_records_failure_and_reraises_without_raw_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")

            def fail(model, prompt, role, effort):
                raise RuntimeError("provider secret=DO_NOT_STORE")

            invoker = TelemetryInvoker(
                invoke=fail,
                store=store,
                task_profile=TaskProfile("security_review", 4, 5, False, False, True),
                plan_id="quality",
                run_id="run-2",
            )
            with self.assertRaisesRegex(RuntimeError, "DO_NOT_STORE"):
                invoker(ModelSpec("provider-a", "model-a", "family-a"), "prompt", "chairman", "high")
            rows = store.summarize()

        self.assertEqual(rows[0].failures, 1)
        self.assertNotEqual(rows[0].model, "DO_NOT_STORE")
        self.assertNotIn("DO_NOT_STORE", str(rows))

    def test_invoker_rejects_run_id_that_cannot_produce_safe_event_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            with self.assertRaisesRegex(ValueError, "run_id"):
                TelemetryInvoker(
                    invoke=lambda *args: "ok",
                    store=store,
                    task_profile=TaskProfile("security_review", 4, 5, False, False, True),
                    plan_id="quality",
                    run_id="r" * 100,
                )

    def test_records_event_without_raw_prompt_or_output_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.db"
            store = TelemetryStore(path, retention_days=90)
            store.record(self._event())

            connection = sqlite3.connect(path)
            try:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(outcome_events)")
                }
                values = connection.execute(
                    "SELECT task_kind, model, outcome, evaluator_score FROM outcome_events"
                ).fetchone()
            finally:
                connection.close()

        self.assertNotIn("prompt", columns)
        self.assertNotIn("output", columns)
        self.assertNotIn("trace", columns)
        self.assertEqual(values, ("security_review", "model-a", "success", 0.9))

    def test_summarizes_performance_by_task_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            store.record(self._event())
            store.record(
                self._event(
                    "event-2",
                    model="model-a",
                    outcome=OutcomeKind.FAILURE,
                    evaluator_score=0.2,
                    latency_ms=800,
                    feedback=FeedbackKind.NEGATIVE,
                    failure_code="timeout",
                )
            )
            summary = store.summarize(task_kind="security_review")

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0].sample_count, 2)
        self.assertEqual(summary[0].successes, 1)
        self.assertEqual(summary[0].failures, 1)
        self.assertEqual(summary[0].unknown_outcomes, 0)
        self.assertEqual(summary[0].positive_feedback, 1)
        self.assertEqual(summary[0].negative_feedback, 1)
        self.assertEqual(summary[0].mean_score, 0.55)
        self.assertEqual(summary[0].mean_latency_ms, 1000.0)
        self.assertEqual(summary[0].mean_execution_calls, 3.0)
        self.assertEqual(summary[0].mean_total_tokens, 800.0)

    def test_records_and_summarizes_final_run_outcomes_separately_from_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            profile = TaskProfile("security_review", 4, 5, False, False, True)
            store.record(self._event())
            store.record_run_outcome(
                event_id="run-outcome-1",
                task_profile=profile,
                plan_id="quality",
                outcome=OutcomeKind.SUCCESS,
                evaluator_score=0.88,
                feedback=FeedbackKind.POSITIVE,
                latency_ms=4500,
                execution_calls=5,
            )
            store.record_run_outcome(
                event_id="run-outcome-2",
                task_profile=profile,
                plan_id="fast",
                outcome=OutcomeKind.FAILURE,
                evaluator_score=0.3,
                feedback=FeedbackKind.NEGATIVE,
                latency_ms=900,
                execution_calls=1,
                failure_code="wrong_answer",
            )
            runs = store.summarize_runs(task_kind="security_review")

        self.assertEqual(len(runs), 2)
        quality = next(item for item in runs if item.plan_id == "quality")
        self.assertEqual(quality.sample_count, 1)
        self.assertEqual(quality.successes, 1)
        self.assertEqual(quality.mean_score, 0.88)
        self.assertEqual(quality.positive_feedback, 1)
        self.assertEqual(quality.mean_execution_calls, 5.0)

    def test_records_final_outcome_from_existing_run_call_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            profile = TaskProfile("decision", 3, 4, False, False, True)
            wrapped = TelemetryInvoker(
                invoke=lambda model, prompt, role, effort: "ok",
                store=store,
                task_profile=profile,
                plan_id="quality",
                run_id="linked-run",
            )
            model = ModelSpec("provider", "model", "family")
            wrapped(model, "prompt", "advisor-1", "high")
            wrapped(model, "prompt", "chairman", "high")

            event_id = store.record_outcome_for_run(
                run_id="linked-run",
                outcome=OutcomeKind.SUCCESS,
                evaluator_score=0.92,
                feedback=FeedbackKind.POSITIVE,
            )
            runs = store.summarize_runs()

        self.assertEqual(event_id, "linked-run:outcome")
        self.assertEqual(runs[0].plan_id, "quality")
        self.assertEqual(runs[0].sample_count, 1)
        self.assertEqual(runs[0].successes, 1)
        self.assertEqual(runs[0].mean_execution_calls, 2.0)

    def test_run_outcome_requires_existing_calls_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            with self.assertRaisesRegex(ValueError, "no telemetry calls"):
                store.record_outcome_for_run(
                    run_id="missing-run",
                    outcome=OutcomeKind.FAILURE,
                    evaluator_score=None,
                    feedback=FeedbackKind.NEGATIVE,
                )

    def test_run_id_underscore_is_not_treated_as_sql_wildcard(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            model = ModelSpec("provider", "model", "family")
            for run_id, plan_id in (("run_1", "quality"), ("runA1", "fast")):
                TelemetryInvoker(
                    invoke=lambda *args: "ok",
                    store=store,
                    task_profile=TaskProfile("decision", 3, 4, False, False, True),
                    plan_id=plan_id,
                    run_id=run_id,
                )(model, "prompt", "actor", "low")

            store.record_outcome_for_run(
                run_id="run_1",
                outcome=OutcomeKind.SUCCESS,
                evaluator_score=0.8,
                feedback=FeedbackKind.POSITIVE,
            )
            runs = store.summarize_runs()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].plan_id, "quality")


    def test_retention_purges_old_events_and_keeps_recent_events(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db", retention_days=30)
            now = datetime(2026, 8, 8, tzinfo=timezone.utc)
            store.record(
                self._event("old", (now - timedelta(days=31)).isoformat()), now=now
            )
            self.assertEqual(store.count_events(), 0)
            store.record(
                self._event("recent", (now - timedelta(days=1)).isoformat()), now=now
            )
            rows = store.count_events()

        self.assertEqual(rows, 1)

    def test_rejects_invalid_feedback_score_and_failure_code(self):
        with self.assertRaises(ValueError):
            self._event(evaluator_score=1.5)
        with self.assertRaises(ValueError):
            self._event(failure_code="not a safe code!")
        with self.assertRaises(ValueError):
            self._event(task_kind="raw user prompt must not be stored")

    def test_integrity_check_and_duplicate_event_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TelemetryStore(Path(directory) / "telemetry.db")
            store.record(self._event())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                store.record(self._event())
            self.assertTrue(store.integrity_check())
            self.assertEqual(store.schema_version(), 2)

    def test_read_only_store_can_summarize_but_cannot_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.db"
            writable = TelemetryStore(path)
            writable.record(self._event())
            readonly = TelemetryStore.open_read_only(path)

            self.assertEqual(readonly.count_events(), 1)
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                readonly.record(self._event("event-2"))

            missing = Path(directory) / "missing.db"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                TelemetryStore.open_read_only(missing)

    def test_migrates_v1_token_column_to_nullable_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.db"
            connection = sqlite3.connect(path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    INSERT INTO schema_meta VALUES ('schema_version', '1');
                    CREATE TABLE outcome_events (
                        event_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL,
                        task_kind TEXT NOT NULL, complexity INTEGER NOT NULL, risk INTEGER NOT NULL,
                        plan_id TEXT NOT NULL, role TEXT NOT NULL, provider TEXT NOT NULL,
                        model TEXT NOT NULL, family TEXT NOT NULL, outcome TEXT NOT NULL,
                        evaluator_score REAL, latency_ms INTEGER NOT NULL,
                        execution_calls INTEGER NOT NULL, total_tokens INTEGER NOT NULL,
                        failure_code TEXT, feedback TEXT NOT NULL, policy_version TEXT NOT NULL
                    );
                    """
                )
                connection.commit()
            finally:
                connection.close()

            store = TelemetryStore(path)
            invoker = TelemetryInvoker(
                invoke=lambda *args: "ok",
                store=store,
                task_profile=TaskProfile("general", 1, 1, False, False, False),
                plan_id="fast",
                run_id="migrated-run",
            )
            invoker(ModelSpec("provider", "model", "family"), "prompt", "actor", "low")
            connection = sqlite3.connect(path)
            try:
                token_value = connection.execute(
                    "SELECT total_tokens FROM outcome_events"
                ).fetchone()[0]
            finally:
                connection.close()
            version = store.schema_version()

        self.assertEqual(version, 2)
        self.assertIsNone(token_value)


if __name__ == "__main__":
    unittest.main()
