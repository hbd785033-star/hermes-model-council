import unittest
from unittest.mock import patch

from model_council.cli import _result_payload
from model_council.decision import DecisionProcess
from model_council.inventory import ModelSpec
from model_council.presets import native_moa_decision_record
from model_council.recommender import Participant, Plan


class HermesNativeAuthorityTests(unittest.TestCase):
    @staticmethod
    def _plan() -> Plan:
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
    def _native_preset(reference_count: int) -> dict[str, object]:
        return {
            "reference_models": [
                {"provider": f"native-{index}", "model": f"model-{index}"}
                for index in range(reference_count)
            ],
            "aggregator": {"provider": "native-aggregator", "model": "aggregator"},
        }

    def test_topology_comes_from_explicit_normalized_preset(self):
        from model_council.hermes_native import adapt_native_moa_outcome

        record = adapt_native_moa_outcome(
            preset="quality",
            normalized_preset=self._native_preset(1),
            decision="native decision",
        )

        self.assertEqual(record.process, DecisionProcess.NATIVE_MOA)
        self.assertEqual(record.preset, "quality")
        self.assertEqual(record.topology_required_calls, 2)
        self.assertIsNone(record.configured_call_ceiling)

    def test_compatibility_wrapper_does_not_reconstruct_plan_topology(self):
        with patch("model_council.presets._preset", side_effect=AssertionError("reconstructed")):
            record = native_moa_decision_record(
                self._plan(),
                normalized_preset=self._native_preset(1),
                decision="native decision",
            )

        self.assertEqual(record.topology_required_calls, 2)

    def test_unknown_observed_usage_stays_unknown(self):
        from model_council.hermes_native import adapt_native_moa_outcome

        record = adapt_native_moa_outcome(
            preset="quality",
            normalized_preset=self._native_preset(3),
            decision="native decision",
        )
        payload = _result_payload(record, probe_call_count=2)

        self.assertEqual(record.topology_required_calls, 4)
        self.assertIsNone(record.observed_calls)
        self.assertIsNone(payload["call_count"])
        self.assertIsNone(payload["execution_call_count"])
        self.assertIsNone(payload["total_call_count"])

    def test_explicit_observed_usage_is_not_normalized_to_topology(self):
        from model_council.hermes_native import adapt_native_moa_outcome

        record = adapt_native_moa_outcome(
            preset="quality",
            normalized_preset=self._native_preset(3),
            decision="native decision",
            observed_calls=7,
        )

        self.assertEqual(record.topology_required_calls, 4)
        self.assertEqual(record.observed_calls, 7)

    def test_malformed_normalized_preset_fails_closed(self):
        from model_council.hermes_native import adapt_native_moa_outcome

        for malformed in ({"aggregator": {}}, {"reference_models": []}):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    adapt_native_moa_outcome(
                        preset="quality",
                        normalized_preset=malformed,
                        decision="native decision",
                    )

    def test_native_and_custom_quality_keep_distinct_process_identity(self):
        from model_council.hermes_native import adapt_native_moa_outcome

        native = adapt_native_moa_outcome(
            preset="quality",
            normalized_preset=self._native_preset(3),
            decision="native decision",
        )

        self.assertEqual(native.process, DecisionProcess.NATIVE_MOA)
        self.assertEqual(self._plan().id, "quality")

    def test_legacy_import_remains_available_but_requires_evidence(self):
        with self.assertRaises(ValueError):
            native_moa_decision_record(self._plan(), decision="native decision")


if __name__ == "__main__":
    unittest.main()
