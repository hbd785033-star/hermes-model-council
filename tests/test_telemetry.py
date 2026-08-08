import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from model_council.telemetry import (
    FeedbackKind,
    OutcomeEvent,
    OutcomeKind,
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
        self.assertEqual(summary[0].mean_score, 0.55)
        self.assertEqual(summary[0].mean_latency_ms, 1000.0)

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
            self.assertEqual(store.schema_version(), 1)


if __name__ == "__main__":
    unittest.main()
