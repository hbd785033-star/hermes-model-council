import unittest

from model_council.activation import (
    ACTIVATION_POLICY_VERSION,
    ActivationDecision,
    recommend_activation,
)
from model_council.analysis import TaskProfile
from model_council.inventory import ModelSpec
from model_council.recommender import Participant, Plan


class ActivationPolicyTests(unittest.TestCase):
    @staticmethod
    def _make_plan(plan_id: str, *, degraded: bool = False) -> Plan:
        model = ModelSpec("provider", f"{plan_id}-model", "family", healthy=True)
        return Plan(
            id=plan_id,
            label=plan_id.title(),
            mode={"fast": "single", "balanced": "moa", "quality": "council"}.get(
                plan_id, "single"
            ),
            participants=(Participant("actor", model, "low"),),
            estimated_calls=1,
            max_calls=1,
            strengths=(),
            risks=(),
            degraded=degraded,
            degradation_reason="fixture_degraded" if degraded else None,
        )

    @classmethod
    def _plans(cls, **degraded: bool) -> list[Plan]:
        return [
            cls._make_plan("fast", degraded=degraded.get("fast", False)),
            cls._make_plan("balanced", degraded=degraded.get("balanced", False)),
            cls._make_plan("quality", degraded=degraded.get("quality", False)),
        ]

    @staticmethod
    def _profile(
        *,
        complexity: int = 1,
        risk: int = 1,
        needs_tools: bool = False,
        needs_freshness: bool = False,
        benefits_from_diversity: bool = False,
    ) -> TaskProfile:
        return TaskProfile(
            kind="general",
            complexity=complexity,
            risk=risk,
            needs_tools=needs_tools,
            needs_freshness=needs_freshness,
            benefits_from_diversity=benefits_from_diversity,
        )

    def test_simple_task_recommends_fast_custom_tool_free(self):
        decision = recommend_activation(self._profile(), self._plans())

        self.assertEqual(
            decision,
            ActivationDecision(
                desired_plan="fast",
                recommended_plan="fast",
                execution_preference="custom_tool_free_ok",
                reasons=("diversity_not_needed",),
            ),
        )
        self.assertEqual(decision.policy_version, ACTIVATION_POLICY_VERSION)
        self.assertEqual(decision.policy_version, "hmc-activation-v1.0")

    def test_moderate_diversity_recommends_balanced(self):
        decision = recommend_activation(
            self._profile(complexity=3, risk=2, benefits_from_diversity=True),
            self._plans(),
        )

        self.assertEqual(decision.desired_plan, "balanced")
        self.assertEqual(decision.recommended_plan, "balanced")
        self.assertEqual(decision.reasons, ("diversity_beneficial",))

    def test_high_risk_diversity_recommends_quality(self):
        decision = recommend_activation(
            self._profile(complexity=2, risk=5, benefits_from_diversity=True),
            self._plans(),
        )

        self.assertEqual(decision.desired_plan, "quality")
        self.assertEqual(decision.recommended_plan, "quality")
        self.assertEqual(decision.reasons, ("diversity_beneficial", "high_risk"))

    def test_high_complexity_diversity_recommends_quality(self):
        decision = recommend_activation(
            self._profile(complexity=5, risk=2, benefits_from_diversity=True),
            self._plans(),
        )

        self.assertEqual(decision.desired_plan, "quality")
        self.assertEqual(decision.reasons, ("diversity_beneficial", "high_complexity"))

    def test_high_risk_and_complexity_reasons_are_unique_and_ordered(self):
        decision = recommend_activation(
            self._profile(complexity=5, risk=5, benefits_from_diversity=True),
            self._plans(),
        )

        self.assertEqual(
            decision.reasons,
            ("diversity_beneficial", "high_risk", "high_complexity"),
        )
        self.assertEqual(len(decision.reasons), len(set(decision.reasons)))

    def test_tools_prefer_native_without_changing_plan_depth(self):
        decision = recommend_activation(
            self._profile(needs_tools=True),
            self._plans(),
        )

        self.assertEqual(decision.desired_plan, "fast")
        self.assertEqual(decision.recommended_plan, "fast")
        self.assertEqual(decision.execution_preference, "hermes_native_preferred")
        self.assertEqual(
            decision.reasons,
            ("diversity_not_needed", "tools_need_native_execution"),
        )

    def test_freshness_prefers_native_without_changing_plan_depth(self):
        decision = recommend_activation(
            self._profile(needs_freshness=True),
            self._plans(),
        )

        self.assertEqual(decision.desired_plan, "fast")
        self.assertEqual(decision.execution_preference, "hermes_native_preferred")
        self.assertEqual(
            decision.reasons,
            ("diversity_not_needed", "freshness_need_native_execution"),
        )

    def test_tools_and_freshness_reasons_are_both_unique_and_ordered(self):
        decision = recommend_activation(
            self._profile(needs_tools=True, needs_freshness=True),
            self._plans(),
        )

        self.assertEqual(
            decision.reasons,
            (
                "diversity_not_needed",
                "tools_need_native_execution",
                "freshness_need_native_execution",
            ),
        )
        self.assertEqual(len(decision.reasons), len(set(decision.reasons)))

    def test_quality_degradation_falls_back_to_balanced(self):
        decision = recommend_activation(
            self._profile(risk=5, benefits_from_diversity=True),
            self._plans(quality=True),
        )

        self.assertEqual(decision.desired_plan, "quality")
        self.assertEqual(decision.recommended_plan, "balanced")
        self.assertEqual(
            decision.reasons,
            (
                "diversity_beneficial",
                "high_risk",
                "desired_plan_degraded",
                "recommendation_fallback",
            ),
        )

    def test_quality_degradation_falls_back_to_fast_when_balanced_is_degraded(self):
        decision = recommend_activation(
            self._profile(risk=5, benefits_from_diversity=True),
            self._plans(quality=True, balanced=True),
        )

        self.assertEqual(decision.desired_plan, "quality")
        self.assertEqual(decision.recommended_plan, "fast")

    def test_balanced_degradation_never_escalates_to_quality(self):
        decision = recommend_activation(
            self._profile(complexity=3, risk=2, benefits_from_diversity=True),
            self._plans(balanced=True),
        )

        self.assertEqual(decision.desired_plan, "balanced")
        self.assertEqual(decision.recommended_plan, "fast")
        self.assertNotEqual(decision.recommended_plan, "quality")
        self.assertEqual(
            decision.reasons,
            (
                "diversity_beneficial",
                "desired_plan_degraded",
                "recommendation_fallback",
            ),
        )

    def test_fast_degraded_keeps_fast_without_claiming_fallback(self):
        decision = recommend_activation(
            self._profile(),
            self._plans(fast=True),
        )

        self.assertEqual(decision.desired_plan, "fast")
        self.assertEqual(decision.recommended_plan, "fast")
        self.assertEqual(
            decision.reasons,
            ("diversity_not_needed", "desired_plan_degraded"),
        )

    def test_balanced_fallback_chain_degraded_does_not_claim_all_plans_degraded(self):
        decision = recommend_activation(
            self._profile(complexity=3, risk=2, benefits_from_diversity=True),
            self._plans(fast=True, balanced=True),
        )

        self.assertEqual(decision.desired_plan, "balanced")
        self.assertEqual(decision.recommended_plan, "fast")
        self.assertEqual(
            decision.reasons,
            (
                "diversity_beneficial",
                "desired_plan_degraded",
                "recommendation_fallback",
            ),
        )

    def test_all_plans_degraded_with_fast_desired_does_not_claim_fallback(self):
        decision = recommend_activation(
            self._profile(),
            self._plans(fast=True, balanced=True, quality=True),
        )

        self.assertEqual(decision.desired_plan, "fast")
        self.assertEqual(decision.recommended_plan, "fast")
        self.assertEqual(
            decision.reasons,
            (
                "diversity_not_needed",
                "desired_plan_degraded",
                "all_candidate_plans_degraded",
            ),
        )

    def test_all_candidates_degraded_returns_fast_with_explicit_truth(self):
        decision = recommend_activation(
            self._profile(risk=5, benefits_from_diversity=True),
            self._plans(fast=True, balanced=True, quality=True),
        )

        self.assertEqual(decision.desired_plan, "quality")
        self.assertEqual(decision.recommended_plan, "fast")
        self.assertEqual(
            decision.reasons,
            (
                "diversity_beneficial",
                "high_risk",
                "desired_plan_degraded",
                "recommendation_fallback",
                "all_candidate_plans_degraded",
            ),
        )

    def test_missing_required_plan_fails_closed(self):
        plans = [self._make_plan("fast"), self._make_plan("quality")]

        with self.assertRaisesRegex(ValueError, "exactly fast, balanced, and quality"):
            recommend_activation(self._profile(), plans)

    def test_duplicate_plan_id_fails_closed(self):
        plans = [
            self._make_plan("fast"),
            self._make_plan("fast"),
            self._make_plan("quality"),
        ]

        with self.assertRaisesRegex(ValueError, "exactly fast, balanced, and quality"):
            recommend_activation(self._profile(), plans)

    def test_unexpected_plan_id_fails_closed(self):
        plans = [
            self._make_plan("fast"),
            self._make_plan("balanced"),
            self._make_plan("experimental"),
        ]

        with self.assertRaisesRegex(ValueError, "exactly fast, balanced, and quality"):
            recommend_activation(self._profile(), plans)

    def test_plan_input_order_does_not_change_decision(self):
        profile = self._profile(risk=5, benefits_from_diversity=True)
        ordered = self._plans(quality=True)
        shuffled = [ordered[2], ordered[0], ordered[1]]

        self.assertEqual(
            recommend_activation(profile, ordered),
            recommend_activation(profile, shuffled),
        )

    def test_non_diversity_signal_remains_authoritative_even_with_high_values(self):
        decision = recommend_activation(
            self._profile(complexity=5, risk=5, benefits_from_diversity=False),
            self._plans(),
        )

        self.assertEqual(decision.desired_plan, "fast")
        self.assertEqual(decision.reasons, ("diversity_not_needed",))


if __name__ == "__main__":
    unittest.main()
