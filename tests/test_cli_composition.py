import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from model_council.analysis import TaskProfile
from model_council.cli import _result_payload, main
from model_council.decision import DecisionProcess, DecisionRecord, DecisionStatus
from model_council.hermes_native import adapt_native_moa_outcome
from model_council.inventory import ModelSpec
from model_council.recommender import Participant, Plan


class CliCompositionTests(unittest.TestCase):
    @staticmethod
    def _fast_plan() -> Plan:
        model = ModelSpec("provider", "model", "family")
        return Plan(
            "fast",
            "Fast",
            "single",
            (Participant("actor", model, "low"),),
            1,
            1,
            (),
            (),
        )

    @staticmethod
    def _quality_plan() -> Plan:
        models = [
            ModelSpec("provider-a", "model-a", "family-a"),
            ModelSpec("provider-b", "model-b", "family-b"),
            ModelSpec("provider-c", "model-c", "family-c"),
            ModelSpec("provider-d", "model-d", "family-d"),
        ]
        return Plan(
            "quality",
            "Quality",
            "council",
            tuple(
                [
                    Participant(f"advisor-{index}", model, "high")
                    for index, model in enumerate(models[:3], start=1)
                ]
                + [Participant("chairman", models[3], "high")]
            ),
            6,
            9,
            (),
            (),
        )

    @staticmethod
    def _completed_fast_record() -> DecisionRecord:
        return DecisionRecord(
            status=DecisionStatus.COMPLETED,
            decision="custom answer",
            process=DecisionProcess.SINGLE,
            preset="fast",
            models_consulted=("provider:model",),
            configured_call_ceiling=1,
            topology_required_calls=1,
            observed_calls=1,
        )

    @staticmethod
    def _recommendation_result(*plans: Plan):
        return (
            TaskProfile("general", 1, 1, False, False, False),
            [],
            list(plans),
            {},
            2,
            1,
        )

    @staticmethod
    def _native_preset() -> dict[str, object]:
        return {
            "reference_models": [
                {"model": "a"},
                {"model": "b"},
                {"model": "c"},
            ],
            "aggregator": {"model": "aggregator"},
        }

    def test_run_wires_approval_recommendation_invoker_runner_and_payload(self):
        fast_plan = self._fast_plan()
        record = self._completed_fast_record()
        sentinel_invoker = object()
        mock_runner = Mock()
        mock_runner.run.return_value = record

        with (
            patch(
                "model_council.cli.prepare_recommendation",
                return_value=self._recommendation_result(fast_plan),
            ) as prepare,
            patch("model_council.cli.HermesInvoker", return_value=sentinel_invoker) as invoker,
            patch("model_council.cli.CouncilRunner", return_value=mock_runner) as runner,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "run",
                        "example task",
                        "--plan",
                        "fast",
                        "--no-probe",
                        "--yes",
                        "--json",
                    ]
                )

        self.assertEqual(result, 0)
        prepare.assert_called_once_with(
            "example task",
            live_probe=False,
            timeout=240,
            refresh_probe=False,
        )
        invoker.assert_called_once_with(timeout=240)
        runner.assert_called_once_with(sentinel_invoker, max_workers=3)
        mock_runner.run.assert_called_once()
        run_task, run_plan = mock_runner.run.call_args.args
        self.assertEqual(run_task, "example task")
        self.assertIs(run_plan, fast_plan)
        self.assertEqual(json.loads(output.getvalue())["decision"], "custom answer")

    def test_run_selects_requested_plan_object_without_recommender_execution(self):
        fast_plan = self._fast_plan()
        quality_plan = self._quality_plan()
        mock_runner = Mock()
        mock_runner.run.return_value = self._completed_fast_record()

        with (
            patch(
                "model_council.cli.prepare_recommendation",
                return_value=self._recommendation_result(fast_plan, quality_plan),
            ),
            patch("model_council.cli.HermesInvoker", return_value=object()),
            patch("model_council.cli.CouncilRunner", return_value=mock_runner),
        ):
            with redirect_stdout(io.StringIO()):
                result = main(
                    ["run", "task", "--plan", "fast", "--no-probe", "--yes", "--json"]
                )

        self.assertEqual(result, 0)
        mock_runner.run.assert_called_once()
        run_task, run_plan = mock_runner.run.call_args.args
        self.assertEqual(run_task, "task")
        self.assertIs(run_plan, fast_plan)
        self.assertIsNot(run_plan, quality_plan)

    def test_run_requires_yes_before_any_composition_dependency(self):
        with (
            patch("model_council.cli.prepare_recommendation") as prepare,
            patch("model_council.cli.HermesInvoker") as invoker,
            patch("model_council.cli.CouncilRunner") as runner,
        ):
            with self.assertRaises(SystemExit):
                main(["run", "task", "--plan", "fast", "--no-probe", "--json"])

        prepare.assert_not_called()
        invoker.assert_not_called()
        runner.assert_not_called()

    def test_json_payload_preserves_legacy_and_decision_contract_fields(self):
        fast_plan = self._fast_plan()
        record = self._completed_fast_record()
        mock_runner = Mock()
        mock_runner.run.return_value = record

        with (
            patch(
                "model_council.cli.prepare_recommendation",
                return_value=self._recommendation_result(fast_plan),
            ),
            patch("model_council.cli.HermesInvoker", return_value=object()),
            patch("model_council.cli.CouncilRunner", return_value=mock_runner),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "run",
                        "example task",
                        "--plan",
                        "fast",
                        "--no-probe",
                        "--yes",
                        "--json",
                    ]
                )
        payload = json.loads(output.getvalue())

        self.assertEqual(result, 0)
        for key in (
            "plan",
            "final",
            "call_count",
            "execution_call_count",
            "total_call_count",
            "status",
            "decision",
            "process",
            "preset",
            "policy_version",
            "models_consulted",
            "configured_call_ceiling",
            "topology_required_calls",
            "observed_calls",
            "fallback_used",
            "fallback_reason",
            "degraded_reasons",
            "warnings",
        ):
            self.assertIn(key, payload)

        self.assertEqual(payload["call_count"], 1)
        self.assertEqual(payload["execution_call_count"], 1)
        self.assertEqual(payload["total_call_count"], 3)
        self.assertEqual(payload["plan"], "fast")
        self.assertEqual(payload["final"], "custom answer")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["decision"], "custom answer")
        self.assertEqual(payload["process"], "single")
        self.assertEqual(payload["preset"], "fast")
        self.assertEqual(payload["policy_version"], "hmc-decision-v1.0")
        self.assertEqual(payload["models_consulted"], ["provider:model"])
        self.assertEqual(payload["configured_call_ceiling"], 1)
        self.assertEqual(payload["topology_required_calls"], 1)
        self.assertEqual(payload["observed_calls"], 1)
        self.assertFalse(payload["fallback_used"])
        self.assertIsNone(payload["fallback_reason"])
        self.assertEqual(payload["degraded_reasons"], [])
        self.assertEqual(payload["warnings"], [])

    def test_native_unknown_usage_composes_through_payload_without_cli(self):
        record = adapt_native_moa_outcome(
            preset="quality",
            normalized_preset=self._native_preset(),
            decision="native decision",
            observed_calls=None,
        )
        payload = _result_payload(record, probe_call_count=2)

        self.assertEqual(payload["process"], "native_moa")
        self.assertEqual(payload["preset"], "quality")
        self.assertIsNone(payload["configured_call_ceiling"])
        self.assertEqual(payload["topology_required_calls"], 4)
        self.assertIsNone(payload["observed_calls"])
        self.assertIsNone(payload["call_count"])
        self.assertIsNone(payload["execution_call_count"])
        self.assertIsNone(payload["total_call_count"])

    def test_native_explicit_usage_composes_through_payload_without_normalization(self):
        record = adapt_native_moa_outcome(
            preset="quality",
            normalized_preset=self._native_preset(),
            decision="native decision",
            observed_calls=7,
        )
        payload = _result_payload(record, probe_call_count=2)

        self.assertEqual(payload["topology_required_calls"], 4)
        self.assertEqual(payload["observed_calls"], 7)
        self.assertEqual(payload["call_count"], 7)
        self.assertEqual(payload["execution_call_count"], 7)
        self.assertEqual(payload["total_call_count"], 9)


if __name__ == "__main__":
    unittest.main()
