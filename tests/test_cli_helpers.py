import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_council.cli import (
    _health_cache_path,
    _merge_health,
    _only_verified,
    _result_payload,
    _store_health_cache,
    _telemetry_path,
    build_parser,
    plan_to_dict,
    probe_candidates,
)
from model_council.health import ProbeResult
from model_council.inventory import ModelSpec
from model_council.recommender import Participant, Plan
from model_council.runner import CouncilResult


class CliHelperTests(unittest.TestCase):
    def test_run_telemetry_is_explicit_opt_in(self):
        parser = build_parser()
        default_args = parser.parse_args(["run", "task", "--plan", "fast", "--yes"])
        opt_in_args = parser.parse_args(
            ["run", "task", "--plan", "fast", "--yes", "--telemetry", "--telemetry-path", "D:/events.db"]
        )

        self.assertFalse(default_args.telemetry)
        self.assertTrue(opt_in_args.telemetry)
        self.assertEqual(opt_in_args.telemetry_path, Path("D:/events.db"))

    def test_record_outcome_parser_requires_run_id_and_validates_choices(self):
        parser = build_parser()
        args = parser.parse_args(
            ["record-outcome", "--run-id", "run-1", "--outcome", "success", "--feedback", "positive"]
        )
        self.assertEqual(args.run_id, "run-1")
        self.assertEqual(args.outcome, "success")
        self.assertEqual(args.feedback, "positive")
        with self.assertRaises(SystemExit):
            parser.parse_args(["record-outcome", "--run-id", "run-1", "--outcome", "maybe"])

    def test_performance_report_parser_is_read_only_and_configurable(self):
        args = build_parser().parse_args(
            ["performance-report", "--task-kind", "security_review",
             "--baseline-plan", "fast", "--minimum-samples", "40", "--json"]
        )
        self.assertEqual(args.task_kind, "security_review")
        self.assertEqual(args.baseline_plan, "fast")
        self.assertEqual(args.minimum_samples, 40)
        self.assertTrue(args.json)

    def test_shadow_router_report_parser_is_advisory_only(self):
        args = build_parser().parse_args(
            ["shadow-router-report", "--task-kind", "security_review",
             "--baseline-plan", "fast", "--minimum-samples", "50", "--json"]
        )
        self.assertEqual(args.task_kind, "security_review")
        self.assertEqual(args.baseline_plan, "fast")
        self.assertEqual(args.minimum_samples, 50)

    def test_telemetry_path_uses_explicit_environment_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MODEL_COUNCIL_TELEMETRY_DIR": directory}
        ):
            self.assertEqual(_telemetry_path(), Path(directory) / "outcomes.db")

    def test_cache_dir_environment_variable_resolves_to_health_cache_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MODEL_COUNCIL_CACHE_DIR": directory}
        ):
            self.assertEqual(
                _health_cache_path(), Path(directory) / "health-cache.json"
            )

    def test_cache_write_failure_is_non_fatal(self):
        class FailingCache:
            def store(self, health):
                raise PermissionError("read-only cache")

        with patch("model_council.cli.print") as mocked_print:
            self.assertFalse(_store_health_cache(FailingCache(), {"model": True}))
        mocked_print.assert_called_once()

    def test_result_payload_discloses_probe_execution_and_total_calls(self):
        result = CouncilResult(
            "quality",
            "ok",
            (("A", "answer"),),
            (),
            ("reviewer-1 failed: timeout",),
            2,
            degraded=True,
            degradation_reason="insufficient_candidates",
            candidate_count=1,
            review_coverage=0.0,
            fallback_source="A",
            task_truncated=True,
        )

        payload = _result_payload(result, probe_call_count=3, probe_cache_hit_count=2)

        self.assertEqual(payload["probe_call_count"], 3)
        self.assertEqual(payload["probe_cache_hit_count"], 2)
        self.assertEqual(payload["execution_call_count"], 2)
        self.assertEqual(payload["total_call_count"], 5)
        self.assertTrue(payload["degraded"])
        self.assertEqual(payload["degradation_reason"], "insufficient_candidates")
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["review_coverage"], 0.0)
        self.assertEqual(payload["fallback_source"], "A")
        self.assertTrue(payload["task_truncated"])

    def test_probe_candidates_are_unique_and_plan_serializes_without_secrets(self):
        sol = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=None)
        claude = ModelSpec("anthropic", "claude-opus-4-6", "anthropic", healthy=None)
        plan = Plan(
            "balanced", "Balanced", "moa",
            (
                Participant("advisor", sol, "medium"),
                Participant("aggregator", claude, "high"),
                Participant("chairman", claude, "high"),
            ),
            2, 3, ("diverse",), ("cost",),
        )

        candidates = probe_candidates([plan])
        payload = plan_to_dict(plan)

        self.assertEqual([model.key for model in candidates], [sol.key, claude.key])
        self.assertEqual(payload["id"], "balanced")
        self.assertEqual(payload["participants"][0]["model"], "gpt-5.6-sol")
        self.assertNotIn("api_key", str(payload).lower())

    def test_all_failed_representatives_disable_the_provider_for_this_run(self):
        failed = ModelSpec("anthropic", "claude-opus-4-8", "anthropic", healthy=False)
        unprobed = ModelSpec("anthropic", "claude-opus-4-7", "anthropic", healthy=None)
        working = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True)
        probe = ProbeResult(
            models=(failed, working),
            diagnostics={failed.key: "session failed", working.key: "ok"},
        )

        merged = _merge_health([failed, unprobed, working], probe)

        self.assertEqual([model.healthy for model in merged], [False, False, True])

    def test_probe_candidates_skip_models_already_checked(self):
        sol = ModelSpec("openai-codex", "gpt-5.6-sol", "openai")
        luna = ModelSpec("openai-codex", "gpt-5.6-luna", "openai")
        plan = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-1", sol, "high"),
                Participant("advisor-2", luna, "high"),
                Participant("chairman", sol, "high"),
            ),
            5,
            9,
            (),
            (),
        )

        candidates = probe_candidates([plan], exclude_keys={sol.key})

        self.assertEqual([model.key for model in candidates], [luna.key])

    def test_live_probe_finalization_rejects_unverified_models(self):
        verified = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True)
        unknown = ModelSpec("openai-codex", "gpt-5.6-terra", "openai", healthy=None)

        finalized = _only_verified([verified, unknown])

        self.assertEqual([model.healthy for model in finalized], [True, False])

    def test_later_failed_sibling_does_not_erase_prior_verified_model(self):
        verified = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True)
        failed_sibling = ModelSpec(
            "openai-codex", "gpt-5.6-sol-pro", "openai", healthy=False
        )
        probe = ProbeResult(
            models=(failed_sibling,),
            diagnostics={failed_sibling.key: "model unavailable"},
        )

        merged = _merge_health([verified, failed_sibling], probe)

        self.assertEqual([model.healthy for model in merged], [True, False])


if __name__ == "__main__":
    unittest.main()
