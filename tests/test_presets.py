import unittest

from model_council.inventory import ModelSpec
from model_council.presets import build_native_moa_config
from model_council.recommender import Participant, Plan


class NativePresetTests(unittest.TestCase):
    @staticmethod
    def _all_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from NativePresetTests._all_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from NativePresetTests._all_keys(item)

    def test_builds_safe_balanced_and_quality_presets_without_losing_existing(self):
        sol = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True)
        claude = ModelSpec("anthropic", "claude-opus-4-6", "anthropic", healthy=True)
        balanced = Plan(
            "balanced", "Balanced", "moa",
            (Participant("advisor", sol, "medium"), Participant("aggregator", claude, "high")),
            2, 3, (), (),
        )
        quality = Plan(
            "quality", "Quality", "council",
            (
                Participant("advisor-1", sol, "high"),
                Participant("advisor-2", claude, "high"),
                Participant("chairman", claude, "high"),
            ),
            5, 9, (), (),
        )
        existing = {"default_preset": "custom", "presets": {"custom": {"enabled": False}}}

        config = build_native_moa_config([balanced, quality], existing)

        self.assertIn("custom", config["presets"])
        self.assertEqual(config["default_preset"], "model-council-balanced")
        self.assertEqual(config["privacy_filter"], "full")
        balanced_preset = config["presets"]["model-council-balanced"]
        self.assertEqual(balanced_preset["aggregator"]["provider"], "anthropic")
        self.assertEqual(balanced_preset["fanout"], "user_turn")
        self.assertEqual(balanced_preset["degraded_reference_policy"], "loud")
        self.assertGreater(balanced_preset["reference_max_tokens"], 0)

    def test_explicit_lens_roles_are_not_serialized_into_native_config(self):
        models = [
            ModelSpec("provider-a", "model-a", "family-a", healthy=True),
            ModelSpec("provider-b", "model-b", "family-b", healthy=True),
            ModelSpec("provider-c", "model-c", "family-c", healthy=True),
            ModelSpec("provider-d", "model-d", "family-d", healthy=True),
        ]
        balanced = Plan(
            "balanced",
            "Balanced",
            "moa",
            (
                Participant("advisor-solution", models[0], "medium"),
                Participant("advisor-risk", models[1], "medium"),
                Participant("aggregator", models[2], "high"),
            ),
            3,
            4,
            (),
            (),
        )
        quality = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-solution", models[0], "high"),
                Participant("advisor-risk", models[1], "high"),
                Participant("advisor-feasibility", models[2], "high"),
                Participant("chairman", models[3], "high"),
            ),
            6,
            9,
            (),
            (),
        )

        config = build_native_moa_config([balanced, quality])
        keys = set(self._all_keys(config))

        for forbidden in ("role", "lens", "lens_id", "lens_policy_version", "prompt"):
            self.assertNotIn(forbidden, keys)
        supported_slot_keys = {"provider", "model", "reasoning_effort", "enabled"}
        for preset in config["presets"].values():
            for slot in [*preset["reference_models"], preset["aggregator"]]:
                self.assertLessEqual(set(slot), supported_slot_keys)


if __name__ == "__main__":
    unittest.main()
