import unittest

from model_council.analysis import TaskProfile
from model_council.inventory import ModelSpec
from model_council.recommender import recommend_plans


class RecommendPlansTests(unittest.TestCase):
    def test_returns_fast_balanced_and_quality_pareto_choices(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", True, True, True, True),
            ModelSpec("openai-codex", "gpt-5.6-luna", "openai", False, True, True, True),
            ModelSpec("anthropic", "claude-fable-5", "anthropic", False, True, False, True),
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic", False, True, True, True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", False, True, False, False),
        ]
        profile = TaskProfile("code", 4, 5, True, False, True)

        plans = recommend_plans(profile, models)

        self.assertEqual([plan.id for plan in plans], ["fast", "balanced", "quality"])
        self.assertEqual(len(plans[0].participants), 1)
        self.assertEqual(plans[1].mode, "moa")
        self.assertGreaterEqual(len({p.model.family for p in plans[1].participants}), 2)
        self.assertFalse(any("fable" in p.model.model for p in plans[1].participants))
        self.assertEqual(plans[2].mode, "council")
        self.assertLessEqual(plans[2].estimated_calls, plans[2].max_calls)
        self.assertFalse(
            any(p.model.provider == "deepseek" for plan in plans for p in plan.participants)
        )

    def test_one_healthy_model_degrades_all_plans_to_single_call(self):
        only = ModelSpec(
            "openai-codex", "gpt-5.6-sol", "openai", healthy=True
        )
        profile = TaskProfile("code", 4, 5, True, False, True)

        plans = recommend_plans(profile, [only])

        self.assertEqual([plan.id for plan in plans], ["fast", "balanced", "quality"])
        for plan in plans:
            self.assertEqual(plan.mode, "single")
            self.assertEqual(plan.estimated_calls, 1)
            self.assertEqual(plan.max_calls, 1)
            self.assertEqual(len(plan.participants), 1)
        self.assertTrue(any("降级" in risk for risk in plans[1].risks))
        self.assertTrue(any("降级" in risk for risk in plans[2].risks))

    def test_rejects_empty_usable_inventory(self):
        models = [ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=False)]
        profile = TaskProfile("general", 1, 1, False, False, False)

        with self.assertRaisesRegex(ValueError, "healthy or unverified"):
            recommend_plans(profile, models)


if __name__ == "__main__":
    unittest.main()
