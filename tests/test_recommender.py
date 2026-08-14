import unittest

from model_council.analysis import TaskProfile
from model_council.inventory import ModelSpec
from model_council.recommender import recommend_plans


class RecommendPlansTests(unittest.TestCase):
    @staticmethod
    def _lens_fixture_models():
        return [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", True, True, True, True),
            ModelSpec("openai-codex", "gpt-5.6-luna", "openai", False, True, True, True),
            ModelSpec("anthropic", "claude-fable-5", "anthropic", False, True, False, True),
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic", False, True, True, True),
            ModelSpec("google", "gemini-3-pro", "google", False, True, True, True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", False, True, False, False),
        ]

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

    def test_tool_task_does_not_claim_custom_runs_have_tools_or_shared_context(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]
        profile = TaskProfile("code", 4, 4, True, False, True)

        plans = recommend_plans(profile, models)

        for plan in plans:
            with self.subTest(plan=plan.id):
                claims = " ".join((*plan.strengths, *plan.risks))
                self.assertNotIn("兼容工具调用和会话上下文", claims)
                self.assertIn("原生 MoA", claims)
                self.assertIn("工具", claims)
                self.assertIn("隔离", claims)

    def test_quality_plan_never_uses_same_model_as_advisor_and_chairman(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("openai-codex", "gpt-5.6-terra", "openai", healthy=True),
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]
        profile = TaskProfile("decision", 5, 5, False, False, True)

        quality = recommend_plans(profile, models)[2]

        advisor_keys = {
            p.model.key for p in quality.participants if p.role.startswith("advisor")
        }
        chairman = quality.chairman
        self.assertNotIn(chairman.model.key, advisor_keys)
        self.assertNotIn("同源", " ".join(quality.risks))
        self.assertIn("不评审自己的候选答案", " ".join(quality.risks))

    def test_quality_reserves_chairman_slot_with_three_models(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]

        advisor_keys = {
            p.model.key for p in quality.participants if p.role.startswith("advisor")
        }
        self.assertNotIn(quality.chairman.model.key, advisor_keys)
        self.assertLessEqual(len(advisor_keys), 2)

    def test_quality_does_not_reuse_same_model_name_through_another_provider(self):
        models = [
            ModelSpec("galaxy-gpt", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
            ModelSpec("ccswitch-claude", "claude-sonnet-4-6", "anthropic", healthy=True),
        ]
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]

        advisor_names = {
            p.model.model.lower()
            for p in quality.participants
            if p.role.startswith("advisor")
        }
        self.assertNotIn(quality.chairman.model.model.lower(), advisor_names)

    def test_quality_advisors_are_unique_by_canonical_model_identity(self):
        models = [
            ModelSpec("provider-a", "shared-model", "family-a", healthy=True),
            ModelSpec("provider-b", "shared-model", "family-b", healthy=True),
            ModelSpec("provider-c", "distinct-model", "family-c", healthy=True),
            ModelSpec("provider-d", "chair-model", "family-d", healthy=True),
        ]

        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]

        advisor_names = [
            p.model.model.casefold()
            for p in quality.participants
            if p.role.startswith("advisor")
        ]
        self.assertEqual(len(advisor_names), len(set(advisor_names)))

    def test_quality_risk_discloses_when_peer_review_is_skipped(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]

        risks = " ".join(quality.risks)
        self.assertIn("跳过互评", risks)
        self.assertNotIn("不评审自己的候选答案", risks)
        self.assertNotIn("匿名互评", " ".join(quality.strengths))

    def test_rejects_empty_usable_inventory(self):
        models = [ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=False)]
        profile = TaskProfile("general", 1, 1, False, False, False)

        with self.assertRaisesRegex(ValueError, "healthy or unverified"):
            recommend_plans(profile, models)

    def test_balanced_assigns_explicit_lenses_without_changing_models_or_budget(self):
        balanced = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True),
            self._lens_fixture_models(),
        )[1]
        advisors = [
            participant
            for participant in balanced.participants
            if participant.role.startswith("advisor")
        ]

        self.assertEqual(
            [participant.role for participant in advisors],
            ["advisor-solution", "advisor-risk"],
        )
        self.assertEqual(
            [participant.model.key for participant in advisors],
            ["openai-codex:gpt-5.6-sol", "google:gemini-3-pro"],
        )
        self.assertEqual(balanced.chairman.role, "aggregator")
        self.assertEqual((balanced.estimated_calls, balanced.max_calls), (3, 4))

    def test_quality_assigns_three_lenses_without_changing_models_or_call_formula(self):
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True),
            self._lens_fixture_models(),
        )[2]
        advisors = [
            participant
            for participant in quality.participants
            if participant.role.startswith("advisor")
        ]

        self.assertEqual(
            [participant.role for participant in advisors],
            ["advisor-solution", "advisor-risk", "advisor-feasibility"],
        )
        self.assertEqual(
            [participant.model.key for participant in advisors],
            [
                "anthropic:claude-fable-5",
                "openai-codex:gpt-5.6-sol",
                "google:gemini-3-pro",
            ],
        )
        self.assertEqual(quality.chairman.role, "chairman")
        self.assertEqual(quality.estimated_calls, 6)

    def test_single_and_degraded_plans_keep_actor_role(self):
        only = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True)

        plans = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True),
            [only],
        )

        self.assertEqual([plan.participants[0].role for plan in plans], ["actor"] * 3)


if __name__ == "__main__":
    unittest.main()
