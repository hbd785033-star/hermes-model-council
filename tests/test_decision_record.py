import unittest

from model_council.cli import _result_payload
from model_council.inventory import ModelSpec
from model_council.presets import native_moa_decision_record
from model_council.recommender import Participant, Plan
from model_council.runner import CouncilRunner


class DecisionRecordTests(unittest.TestCase):
    def test_single_invocation_failure_returns_failed_decision_record(self):
        model = ModelSpec("provider-a", "model-a", "family-a")
        plan = Plan(
            "fast",
            "Fast",
            "single",
            (Participant("actor", model, "low"),),
            1,
            1,
            (),
            (),
        )

        def invoke(*_args):
            raise RuntimeError("timed out")

        record = CouncilRunner(invoke).run("task", plan)

        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.decision)
        self.assertEqual(record.process, "single")
        self.assertEqual(record.configured_call_ceiling, 1)
        self.assertEqual(record.topology_required_calls, 1)
        self.assertEqual(record.observed_calls, 1)

    def test_empty_single_response_returns_failed_decision_record(self):
        model = ModelSpec("provider-a", "model-a", "family-a")
        plan = Plan(
            "fast",
            "Fast",
            "single",
            (Participant("actor", model, "low"),),
            1,
            1,
            (),
            (),
        )

        record = CouncilRunner(lambda *_args: "").run("task", plan)

        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.decision)
        self.assertEqual(record.degraded_reasons, ("single_invocation_failed",))
        self.assertEqual(record.warnings, ("actor failed: empty_response",))

    def test_all_moa_advisors_fail_returns_failed_decision_record(self):
        models = [
            ModelSpec("provider-a", "model-a", "family-a"),
            ModelSpec("provider-b", "model-b", "family-b"),
            ModelSpec("provider-c", "model-c", "family-c"),
        ]
        plan = Plan(
            "balanced",
            "Balanced",
            "moa",
            (
                Participant("advisor-1", models[0], "medium"),
                Participant("advisor-2", models[1], "medium"),
                Participant("aggregator", models[2], "high"),
            ),
            3,
            4,
            (),
            (),
        )

        record = CouncilRunner(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("unavailable")),
            max_workers=1,
        ).run("task", plan)

        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.decision)
        self.assertEqual(record.process, "custom_moa")
        self.assertEqual(record.topology_required_calls, 3)
        self.assertEqual(record.observed_calls, 2)
        self.assertEqual(record.degraded_reasons, ("all_moa_advisors_failed",))

    def test_all_custom_council_advisors_fail_returns_failed_decision_record(self):
        models = [
            ModelSpec("provider-a", "model-a", "family-a"),
            ModelSpec("provider-b", "model-b", "family-b"),
            ModelSpec("provider-c", "model-c", "family-c"),
        ]
        plan = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("chairman", models[2], "high"),
            ),
            5,
            9,
            (),
            (),
        )

        record = CouncilRunner(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("unavailable")),
            max_workers=1,
        ).run("task", plan)

        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.decision)
        self.assertEqual(record.process, "custom_council")
        self.assertEqual(record.topology_required_calls, 5)
        self.assertEqual(record.observed_calls, 2)
        self.assertEqual(record.degraded_reasons, ("all_council_advisors_failed",))

    def test_programming_error_is_not_converted_to_failed_decision_record(self):
        model = ModelSpec("provider-a", "model-a", "family-a")
        plan = Plan(
            "fast",
            "Fast",
            "single",
            (Participant("actor", model, "low"),),
            1,
            1,
            (),
            (),
        )

        with self.assertRaisesRegex(ValueError, "programming defect"):
            CouncilRunner(
                lambda *_args: (_ for _ in ()).throw(ValueError("programming defect"))
            ).run("task", plan)

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
            (
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("advisor-3", models[2], "high"),
                Participant("chairman", models[3], "high"),
            ),
            6,
            9,
            (),
            (),
        )

    def test_completed_custom_council_records_process_models_and_budget_dimensions(self):
        plan = self._quality_plan()

        def invoke(_model, _prompt, role, _effort):
            if role.startswith("advisor"):
                return f"proposal from {role}"
            if role.startswith("reviewer"):
                return "review"
            return "chairman decision"

        record = CouncilRunner(invoke, max_workers=1).run("task", plan, seed=7)

        self.assertEqual(record.status, "completed")
        self.assertEqual(record.decision, "chairman decision")
        self.assertEqual(record.preset, "quality")
        self.assertEqual(record.process, "custom_council")
        self.assertEqual(
            record.models_consulted,
            ("model-a", "model-b", "model-c", "model-d"),
        )
        self.assertEqual(record.policy_version, "hmc-decision-v1.0")
        self.assertEqual(record.configured_call_ceiling, 9)
        self.assertEqual(record.topology_required_calls, 6)
        self.assertEqual(record.observed_calls, 6)
        self.assertFalse(record.fallback_used)
        self.assertEqual(record.degraded_reasons, ())

    def test_partial_advisor_failure_returns_degraded_decision_record(self):
        plan = self._quality_plan()

        def invoke(_model, _prompt, role, _effort):
            if role == "advisor-1":
                raise RuntimeError("unavailable")
            if role.startswith("advisor"):
                return f"proposal from {role}"
            if role.startswith("reviewer"):
                return "review"
            return "chairman decision"

        record = CouncilRunner(invoke, max_workers=1).run("task", plan, seed=7)

        self.assertEqual(record.status, "degraded")
        self.assertEqual(record.decision, "chairman decision")
        self.assertEqual(record.degraded_reasons, ("participant_failure",))
        self.assertEqual(record.configured_call_ceiling, 9)
        self.assertEqual(record.topology_required_calls, 6)
        self.assertEqual(record.observed_calls, 6)

    def test_chairman_failure_records_non_ranked_candidate_fallback(self):
        plan = self._quality_plan()

        def invoke(_model, _prompt, role, _effort):
            if role.startswith("advisor"):
                return f"proposal from {role}"
            if role.startswith("reviewer"):
                return "review"
            raise RuntimeError("chairman unavailable")

        record = CouncilRunner(invoke, max_workers=1).run("task", plan, seed=7)

        self.assertEqual(record.status, "degraded")
        self.assertIsNotNone(record.decision)
        self.assertTrue(record.fallback_used)
        self.assertEqual(
            record.fallback_reason,
            "chairman_failed_candidate_fallback",
        )
        self.assertNotIn("best", record.fallback_reason)
        self.assertNotIn("winner", record.fallback_reason)
        self.assertNotIn("highest", record.fallback_reason)
        self.assertEqual(record.observed_calls, 6)

    def test_quality_preset_has_distinct_native_and_custom_process_identity(self):
        plan = self._quality_plan()

        native_record = native_moa_decision_record(
            plan,
            normalized_preset={
                "reference_models": [{"model": "a"}, {"model": "b"}, {"model": "c"}],
                "aggregator": {"model": "aggregator"},
            },
            decision="native aggregator decision",
            models_consulted=("model-a", "model-b", "model-c", "model-d"),
        )

        custom_record = CouncilRunner(
            lambda _model, _prompt, role, _effort: (
                "review" if role.startswith("reviewer") else "custom decision"
            ),
            max_workers=1,
        ).run("task", plan, seed=7)

        self.assertEqual(native_record.preset, "quality")
        self.assertEqual(native_record.process, "native_moa")
        self.assertEqual(native_record.status, "completed")
        self.assertEqual(
            native_record.models_consulted,
            ("model-a", "model-b", "model-c", "model-d"),
        )
        self.assertIsNone(native_record.configured_call_ceiling)
        self.assertEqual(native_record.topology_required_calls, 4)
        self.assertIsNone(native_record.observed_calls)

        self.assertEqual(custom_record.preset, "quality")
        self.assertEqual(custom_record.process, "custom_council")
        self.assertEqual(custom_record.topology_required_calls, 6)

    def test_public_payload_exposes_decision_contract_and_budget_provenance(self):
        plan = self._quality_plan()

        record = CouncilRunner(
            lambda _model, _prompt, role, _effort: (
                "review" if role.startswith("reviewer") else "decision"
            ),
            max_workers=1,
        ).run("task", plan, seed=7)

        payload = _result_payload(record)

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["decision"], "decision")
        self.assertEqual(payload["process"], "custom_council")
        self.assertEqual(payload["preset"], "quality")
        self.assertEqual(payload["policy_version"], "hmc-decision-v1.0")
        self.assertEqual(payload["models_consulted"], ["model-a", "model-b", "model-c", "model-d"])
        self.assertEqual(payload["configured_call_ceiling"], 9)
        self.assertEqual(payload["topology_required_calls"], 6)
        self.assertEqual(payload["observed_calls"], 6)

    def test_legacy_payload_does_not_guess_process_from_preset(self):
        from model_council.decision import CouncilResult

        for preset in ("fast", "balanced", "quality"):
            with self.subTest(preset=preset):
                payload = _result_payload(CouncilResult(preset, "legacy", (), (), (), 1))
                self.assertIsNone(payload["process"])

    def test_requested_moa_without_advisors_records_actual_single_process(self):
        model = ModelSpec("provider-a", "model-a", "family-a")
        plan = Plan(
            "balanced",
            "Balanced",
            "moa",
            (Participant("aggregator", model, "high"),),
            1,
            3,
            (),
            (),
        )

        record = CouncilRunner(lambda *_args: "single fallback").run("task", plan)

        self.assertEqual(record.preset, "balanced")
        self.assertEqual(record.process, "single")
        self.assertEqual(record.status, "degraded")
        self.assertEqual(record.degraded_reasons, ("no_advisors",))

    def test_decision_record_rejects_contradictory_terminal_states(self):
        from model_council.decision import (
            DecisionProcess,
            DecisionRecord,
            DecisionStatus,
        )

        common = {
            "process": DecisionProcess.SINGLE,
            "preset": "fast",
            "models_consulted": (),
            "configured_call_ceiling": 1,
            "topology_required_calls": 1,
            "observed_calls": 1,
        }
        invalid = (
            (DecisionStatus.COMPLETED, None),
            (DecisionStatus.COMPLETED, "   "),
            (DecisionStatus.DEGRADED, None),
            (DecisionStatus.FAILED, "answer"),
        )

        for status, decision in invalid:
            with self.subTest(status=status, decision=decision):
                with self.assertRaises(ValueError):
                    DecisionRecord(status=status, decision=decision, **common)

        DecisionRecord(status=DecisionStatus.COMPLETED, decision="answer", **common)
        DecisionRecord(status=DecisionStatus.DEGRADED, decision="answer", **common)
        DecisionRecord(status=DecisionStatus.FAILED, decision=None, **common)

    def test_unknown_native_observed_usage_stays_unknown_in_public_json(self):
        plan = self._quality_plan()
        record = native_moa_decision_record(
            plan,
            normalized_preset={
                "reference_models": [{"model": "a"}, {"model": "b"}, {"model": "c"}],
                "aggregator": {"model": "aggregator"},
            },
            decision="native decision",
            models_consulted=("model-a", "model-b", "model-c", "model-d"),
        )

        payload = _result_payload(record, probe_call_count=2)

        self.assertIsNone(payload["observed_calls"])
        self.assertIsNone(payload["call_count"])
        self.assertIsNone(payload["execution_call_count"])
        self.assertIsNone(payload["total_call_count"])


if __name__ == "__main__":
    unittest.main()