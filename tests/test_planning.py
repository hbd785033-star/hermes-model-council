from __future__ import annotations

import dataclasses
import unittest

from model_council.activation import ACTIVATION_POLICY_VERSION
from model_council.analysis import TaskProfile
from model_council.inventory import ModelSpec
from model_council.lenses import LENS_POLICY_VERSION
from model_council.planning import (
    HMC_PLANNER_CONTRACT_VERSION,
    PlannerCandidate,
    PlannerRecommendation,
    PlannerRequest,
    plan_task,
)
from model_council.recommender import Participant, Plan, recommend_plans


class TypedPlannerContractTests(unittest.TestCase):
    @staticmethod
    def _model(name: str = "model") -> ModelSpec:
        return ModelSpec("provider", name, "family", healthy=True)

    @classmethod
    def _plan(
        cls,
        plan_id: str,
        *,
        mode: str | None = None,
        roles: tuple[str, ...] | None = None,
        estimated_calls: int = 1,
        max_calls: int = 1,
        degraded: bool = False,
        degradation_reason: str | None = None,
    ) -> Plan:
        default_mode = {"fast": "single", "balanced": "moa", "quality": "council"}
        default_roles = {
            "fast": ("actor",),
            "balanced": ("advisor-solution", "advisor-risk", "aggregator"),
            "quality": (
                "advisor-solution",
                "advisor-risk",
                "advisor-feasibility",
                "chairman",
            ),
        }
        selected_roles = roles if roles is not None else default_roles.get(plan_id, ("actor",))
        return Plan(
            id=plan_id,
            label=plan_id.title(),
            mode=mode if mode is not None else default_mode.get(plan_id, "single"),
            participants=tuple(
                Participant(role, cls._model(f"model-{index}"), "low")
                for index, role in enumerate(selected_roles)
            ),
            estimated_calls=estimated_calls,
            max_calls=max_calls,
            strengths=(),
            risks=(),
            degraded=degraded,
            degradation_reason=degradation_reason,
        )

    @classmethod
    def _plans(cls, **degraded: bool) -> list[Plan]:
        return [
            cls._plan("fast", estimated_calls=1, max_calls=1),
            cls._plan(
                "balanced",
                estimated_calls=3,
                max_calls=4,
                degraded=degraded.get("balanced", False),
                degradation_reason="balanced_degraded" if degraded.get("balanced", False) else None,
            ),
            cls._plan(
                "quality",
                estimated_calls=6,
                max_calls=9,
                degraded=degraded.get("quality", False),
                degradation_reason="quality_degraded" if degraded.get("quality", False) else None,
            ),
        ]

    @staticmethod
    def _request(
        *,
        complexity: int = 1,
        risk: int = 1,
        needs_tools: bool = False,
        needs_freshness: bool = False,
        diversity: bool = False,
    ) -> PlannerRequest:
        return PlannerRequest(
            TaskProfile(
                kind="general",
                complexity=complexity,
                risk=risk,
                needs_tools=needs_tools,
                needs_freshness=needs_freshness,
                benefits_from_diversity=diversity,
            )
        )

    def test_public_contracts_are_frozen_and_versioned(self) -> None:
        request = self._request()
        candidate = PlannerCandidate(
            id="fast",
            mode="single",
            estimated_calls=1,
            max_calls=1,
            degraded=False,
            degradation_reason=None,
            planned_lens_ids=(),
        )
        recommendation = plan_task(request, self._plans())

        self.assertEqual(HMC_PLANNER_CONTRACT_VERSION, "hmc-planner-v1.0")
        self.assertEqual(request.contract_version, HMC_PLANNER_CONTRACT_VERSION)
        self.assertEqual(
            recommendation.planner_contract_version,
            HMC_PLANNER_CONTRACT_VERSION,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.contract_version = "changed"  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            candidate.mode = "moa"  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            recommendation.recommended_plan = "quality"  # type: ignore[misc]

    def test_fast_balanced_and_quality_recommendations_use_activation_policy(self) -> None:
        cases = (
            (self._request(), "fast"),
            (self._request(complexity=3, risk=2, diversity=True), "balanced"),
            (self._request(risk=5, diversity=True), "quality"),
        )
        for request, expected in cases:
            with self.subTest(expected=expected):
                recommendation = plan_task(request, self._plans())
                self.assertEqual(recommendation.desired_plan, expected)
                self.assertEqual(recommendation.recommended_plan, expected)

    def test_candidate_order_does_not_change_recommendation(self) -> None:
        request = self._request(risk=5, diversity=True)
        plans = self._plans()

        self.assertEqual(
            plan_task(request, plans),
            plan_task(request, [plans[2], plans[0], plans[1]]),
        )

    def test_candidate_ids_must_be_exactly_fast_balanced_quality(self) -> None:
        cases = (
            self._plans()[:2],
            [self._plans()[0], self._plans()[0], self._plans()[2]],
            [self._plans()[0], self._plans()[1], self._plan("experimental")],
            [*self._plans(), self._plan("experimental")],
        )
        for plans in cases:
            with self.subTest(ids=[plan.id for plan in plans]):
                with self.assertRaisesRegex(ValueError, "exactly fast, balanced, and quality"):
                    plan_task(self._request(), plans)

    def test_candidate_mode_and_call_numbers_fail_closed(self) -> None:
        invalid_balanced = (
            self._plan("balanced", mode="runtime"),
            self._plan("balanced", estimated_calls=0, max_calls=1),
            self._plan("balanced", estimated_calls=1, max_calls=0),
            self._plan("balanced", estimated_calls=True, max_calls=1),
            self._plan("balanced", estimated_calls=2, max_calls=1),
        )
        for balanced in invalid_balanced:
            plans = self._plans()
            plans[1] = balanced
            with self.subTest(candidate=balanced):
                with self.assertRaises(ValueError):
                    plan_task(self._request(), plans)

    def test_planner_candidate_rejects_malformed_lens_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "planned_lens_ids"):
            PlannerCandidate(
                id="balanced",
                mode="moa",
                estimated_calls=2,
                max_calls=3,
                degraded=False,
                degradation_reason=None,
                planned_lens_ids=("solution", "invented"),
            )

    def test_degraded_desired_plan_falls_downward_and_recommended_candidate_owns_fields(self) -> None:
        recommendation = plan_task(
            self._request(risk=5, diversity=True),
            self._plans(quality=True),
        )

        self.assertEqual(recommendation.desired_plan, "quality")
        self.assertEqual(recommendation.recommended_plan, "balanced")
        self.assertEqual(recommendation.selected_plan_mode, "moa")
        self.assertEqual(recommendation.planned_call_count, 3)
        self.assertEqual(recommendation.planner_call_ceiling, 4)
        self.assertFalse(recommendation.degraded)
        self.assertIsNone(recommendation.degradation_reason)
        self.assertEqual(recommendation.planned_lens_ids, ("solution", "risk"))

    def test_planned_lenses_come_from_actual_advisor_roles(self) -> None:
        plans = self._plans()
        plans[1] = self._plan(
            "balanced",
            roles=("advisor-feasibility", "aggregator"),
            estimated_calls=2,
            max_calls=3,
        )

        recommendation = plan_task(
            self._request(complexity=3, diversity=True),
            plans,
        )

        self.assertEqual(recommendation.planned_lens_ids, ("feasibility",))

    def test_actor_aggregator_and_chairman_are_not_lens_ids(self) -> None:
        recommendation = plan_task(self._request(), self._plans())

        self.assertEqual(recommendation.selected_plan_mode, "single")
        self.assertEqual(recommendation.planned_lens_ids, ())

    def test_policy_provenance_is_real_and_has_no_fake_recommender_version(self) -> None:
        recommendation = plan_task(self._request(), self._plans())

        self.assertEqual(recommendation.activation_policy_version, ACTIVATION_POLICY_VERSION)
        self.assertEqual(recommendation.activation_policy_version, "hmc-activation-v1.0")
        self.assertEqual(recommendation.lens_policy_version, LENS_POLICY_VERSION)
        self.assertEqual(recommendation.lens_policy_version, "hmc-lenses-v1.0")
        self.assertNotIn("recommender_policy_version", recommendation.__dataclass_fields__)

    def test_contract_has_planning_truth_only_without_model_or_runtime_leakage(self) -> None:
        candidate_fields = set(PlannerCandidate.__dataclass_fields__)
        recommendation_fields = set(PlannerRecommendation.__dataclass_fields__)
        forbidden = {
            "participant",
            "participants",
            "model",
            "models",
            "provider",
            "family",
            "runtime",
            "runtime_id",
            "executor",
            "reviewer",
            "fallback_runtime",
            "workspace",
            "worktree",
            "runtime_health",
            "capability_status",
            "observed_calls",
            "models_consulted",
            "execution_status",
            "execution_result",
            "approval_status",
            "verification_status",
            "verified",
            "evaluation_score",
            "confidence_as_truth",
            "native_execution",
            "executed_lenses",
        }

        self.assertFalse(candidate_fields & forbidden)
        self.assertFalse(recommendation_fields & forbidden)

    def test_tools_and_freshness_change_preference_without_selecting_runtime(self) -> None:
        for request in (
            self._request(needs_tools=True),
            self._request(needs_freshness=True),
        ):
            with self.subTest(profile=request.task_profile):
                recommendation = plan_task(request, self._plans())
                self.assertEqual(
                    recommendation.execution_preference,
                    "hermes_native_preferred",
                )
                self.assertEqual(recommendation.recommended_plan, "fast")
                self.assertNotIn("executor", recommendation.__dataclass_fields__)
                self.assertNotIn("runtime", recommendation.__dataclass_fields__)

    def test_invalid_request_contract_version_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract_version"):
            PlannerRequest(
                self._request().task_profile,
                contract_version="hmc-planner-v2.0",
            )

    def test_full_chain_uses_real_recommender_plans_and_role_lenses(self) -> None:
        models = [
            ModelSpec("openai", "gpt-sol", "openai", healthy=True),
            ModelSpec("anthropic", "claude-opus", "anthropic", healthy=True),
            ModelSpec("google", "gemini-pro", "google", healthy=True),
            ModelSpec("deepseek", "deepseek-pro", "deepseek", healthy=True),
        ]
        request = self._request(risk=5, diversity=True)
        plans = recommend_plans(request.task_profile, models)

        recommendation = plan_task(request, plans)
        quality = next(plan for plan in plans if plan.id == "quality")
        advisor_roles = tuple(
            participant.role
            for participant in quality.participants
            if participant.role.startswith("advisor")
        )

        self.assertEqual(recommendation.recommended_plan, "quality")
        self.assertEqual(advisor_roles, (
            "advisor-solution",
            "advisor-risk",
            "advisor-feasibility",
        ))
        self.assertEqual(recommendation.planned_lens_ids, (
            "solution",
            "risk",
            "feasibility",
        ))
        self.assertEqual(recommendation.planned_call_count, quality.estimated_calls)
        self.assertEqual(recommendation.planner_call_ceiling, quality.max_calls)

    def test_repeated_planning_is_deterministic(self) -> None:
        request = self._request(complexity=3, diversity=True)
        plans = self._plans()

        self.assertEqual(plan_task(request, plans), plan_task(request, plans))


if __name__ == "__main__":
    unittest.main()
