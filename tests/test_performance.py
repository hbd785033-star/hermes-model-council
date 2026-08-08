import unittest

from model_council.performance import build_performance_report
from model_council.telemetry import RunPerformanceSummary


class PerformanceReportTests(unittest.TestCase):
    def _summary(
        self, plan_id, samples, successes, failures, *, score, latency, unknown=0
    ):
        return RunPerformanceSummary(
            task_kind="security_review",
            plan_id=plan_id,
            sample_count=samples,
            successes=successes,
            failures=failures,
            unknown_outcomes=unknown,
            positive_feedback=successes,
            negative_feedback=failures,
            mean_score=score,
            mean_latency_ms=latency,
            mean_execution_calls=1.0 if plan_id == "fast" else 5.0,
            mean_total_tokens=None,
        )

    def test_reports_metric_specific_regret_without_composite_winner(self):
        summaries = (
            self._summary("fast", 40, 28, 12, score=0.70, latency=1000.0),
            self._summary("quality", 40, 34, 6, score=0.85, latency=5000.0),
            self._summary("balanced", 10, 8, 2, score=0.80, latency=2500.0),
        )

        report = build_performance_report(
            summaries,
            task_kind="security_review",
            baseline_plan="fast",
            minimum_samples=30,
        )

        self.assertTrue(report.ready)
        self.assertFalse(hasattr(report, "recommended_plan"))
        fast = next(item for item in report.plans if item.plan_id == "fast")
        quality = next(item for item in report.plans if item.plan_id == "quality")
        balanced = next(item for item in report.plans if item.plan_id == "balanced")
        self.assertAlmostEqual(fast.success_regret, 0.15)
        self.assertAlmostEqual(fast.score_regret, 0.15)
        self.assertEqual(fast.latency_regret_ms, 0.0)
        self.assertEqual(quality.success_regret, 0.0)
        self.assertEqual(quality.score_regret, 0.0)
        self.assertEqual(quality.latency_regret_ms, 4000.0)
        self.assertFalse(balanced.eligible)
        self.assertIsNone(balanced.success_regret)
        self.assertGreater(quality.success_ci_high, quality.success_rate)
        self.assertLess(quality.success_ci_low, quality.success_rate)

    def test_unknown_outcomes_do_not_count_as_success_or_failure(self):
        summary = self._summary(
            "fast", 40, 20, 10, score=0.6, latency=1000.0, unknown=10
        )

        report = build_performance_report(
            (summary,), task_kind="security_review", minimum_samples=30
        )

        self.assertAlmostEqual(report.plans[0].success_rate, 20 / 30)
        self.assertEqual(report.plans[0].known_outcomes, 30)

    def test_report_stays_not_ready_when_baseline_or_comparison_is_undersampled(self):
        report = build_performance_report(
            (
                self._summary("fast", 20, 15, 5, score=0.7, latency=900.0),
                self._summary("quality", 40, 35, 5, score=0.9, latency=4500.0),
            ),
            task_kind="security_review",
            baseline_plan="fast",
            minimum_samples=30,
        )

        self.assertFalse(report.ready)
        self.assertTrue(any("baseline" in warning for warning in report.warnings))
        self.assertTrue(all(item.success_regret is None for item in report.plans))

    def test_rejects_duplicate_plan_summaries_and_invalid_sample_threshold(self):
        summary = self._summary("fast", 40, 30, 10, score=0.8, latency=1000.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_performance_report((summary, summary), task_kind="security_review")
        with self.assertRaisesRegex(ValueError, "minimum_samples"):
            build_performance_report((summary,), task_kind="security_review", minimum_samples=0)

    def test_json_payload_contains_metrics_and_warnings_only(self):
        report = build_performance_report(
            (self._summary("fast", 5, 4, 1, score=0.8, latency=1000.0),),
            task_kind="security_review",
            minimum_samples=30,
        )

        payload = report.to_dict()

        self.assertFalse(payload["ready"])
        self.assertIn("warnings", payload)
        self.assertNotIn("recommended_plan", payload)
        self.assertEqual(payload["plans"][0]["plan_id"], "fast")


if __name__ == "__main__":
    unittest.main()
