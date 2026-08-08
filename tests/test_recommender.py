import unittest

from model_council.analysis import TaskProfile
from model_council.inventory import ModelSpec
from model_council.recommender import recommend_plans
from model_council.runner import CouncilRunner


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

    def test_quality_risk_describes_cross_role_bias_after_self_exclusion(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]

        risks = " ".join(quality.risks)
        self.assertIn("不评审自己的候选答案", risks)
        self.assertNotIn("评审者与候选答案可能重叠", risks)

    def test_two_model_quality_plan_does_not_advertise_or_budget_empty_peer_review(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]

        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]

        self.assertEqual(quality.estimated_calls, 2)
        self.assertNotIn("匿名互评", " ".join(quality.strengths))
        self.assertIn("跳过 Peer Review", " ".join(quality.risks))

    def test_two_model_recommended_quality_executes_only_advisor_and_chairman(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True),
        ]
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]
        roles = []

        def invoke(model, prompt, role, effort):
            roles.append(role)
            return role

        result = CouncilRunner(invoke=invoke, max_workers=1).run("task", quality)

        self.assertEqual(roles, ["advisor-1", "chairman"])
        self.assertEqual(result.call_count, quality.estimated_calls)
        self.assertTrue(result.degraded)
        self.assertEqual(result.degradation_reason, "insufficient_candidates")

    def test_rejects_empty_usable_inventory(self):
        models = [ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=False)]
        profile = TaskProfile("general", 1, 1, False, False, False)

        with self.assertRaisesRegex(ValueError, "healthy or unverified"):
            recommend_plans(profile, models)


if __name__ == "__main__":
    unittest.main()
