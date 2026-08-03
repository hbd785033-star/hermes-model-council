import unittest

from model_council.inventory import ModelSpec
from model_council.presets import build_native_moa_config
from model_council.recommender import Participant, Plan


class NativePresetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
