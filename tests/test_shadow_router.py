import unittest

from model_council.performance import build_performance_report
from model_council.shadow_router import build_shadow_router_report
from model_council.telemetry import RunPerformanceSummary


class ShadowRouterReportTests(unittest.TestCase):
    def _summary(self, plan, n, success, score, latency):
        return RunPerformanceSummary(
            task_kind="security_review",
            plan_id=plan,
            sample_count=n,
            successes=success,
            failures=n - success,
            unknown_outcomes=0,
            positive_feedback=success,
            negative_feedback=n - success,
            mean_score=score,
            mean_latency_ms=latency,
            mean_execution_calls=1.0 if plan == "fast" else 5.0,
            mean_total_tokens=None,
        )

    def test_proposes_candidate_only_for_non_overlapping_quality_gain(self):
        performance = build_performance_report(
            (
                self._summary("fast", 100, 60, 0.60, 1000.0),
                self._summary("quality", 100, 90, 0.90, 5000.0),
            ),
            task_kind="security_review",
            minimum_samples=30,
        )

        report = build_shadow_router_report(performance)

        self.assertTrue(report.ready)
        self.assertFalse(report.apply_automatically)
        self.assertEqual(len(report.proposals), 1)
        proposal = report.proposals[0]
        self.assertEqual(proposal.baseline_plan, "fast")
        self.assertEqual(proposal.candidate_plan, "quality")
        self.assertGreater(
            proposal.candidate_success_ci_low, proposal.baseline_success_ci_high
        )
        self.assertEqual(proposal.score_delta, 0.3)
        self.assertEqual(proposal.latency_delta_ms, 4000.0)
        self.assertTrue(proposal.rollback_conditions)

    def test_overlapping_intervals_produce_no_route_proposal(self):
        performance = build_performance_report(
            (
                self._summary("fast", 40, 28, 0.70, 1000.0),
                self._summary("quality", 40, 32, 0.80, 4500.0),
            ),
            task_kind="security_review",
            minimum_samples=30,
        )

        report = build_shadow_router_report(performance)

        self.assertTrue(report.ready)
        self.assertEqual(report.proposals, ())
        self.assertTrue(any("overlap" in warning for warning in report.warnings))

    def test_score_regression_blocks_proposal_even_with_success_gain(self):
        performance = build_performance_report(
            (
                self._summary("fast", 100, 60, 0.80, 1000.0),
                self._summary("quality", 100, 90, 0.70, 5000.0),
            ),
            task_kind="security_review",
            minimum_samples=30,
        )

        report = build_shadow_router_report(performance)

        self.assertEqual(report.proposals, ())
        self.assertTrue(any("score" in warning for warning in report.warnings))

    def test_not_ready_performance_cannot_propose(self):
        performance = build_performance_report(
            (
                self._summary("fast", 10, 8, 0.8, 1000.0),
                self._summary("quality", 10, 9, 0.9, 5000.0),
            ),
            task_kind="security_review",
            minimum_samples=30,
        )

        report = build_shadow_router_report(performance)

        self.assertFalse(report.ready)
        self.assertEqual(report.proposals, ())

    def test_payload_has_no_auto_apply_or_config_mutation(self):
        performance = build_performance_report(
            (
                self._summary("fast", 100, 60, 0.60, 1000.0),
                self._summary("quality", 100, 90, 0.90, 5000.0),
            ),
            task_kind="security_review",
            minimum_samples=30,
        )

        payload = build_shadow_router_report(performance).to_dict()

        self.assertFalse(payload["apply_automatically"])
        self.assertNotIn("config_patch", payload)
        self.assertNotIn("router_weights", payload)


if __name__ == "__main__":
    unittest.main()
