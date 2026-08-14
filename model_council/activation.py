"""Deterministic activation policy for HMC plan recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .analysis import TaskProfile
from .recommender import Plan

ACTIVATION_POLICY_VERSION = "hmc-activation-v1.0"

PlanId = Literal["fast", "balanced", "quality"]
ExecutionPreference = Literal[
    "custom_tool_free_ok",
    "hermes_native_preferred",
]


@dataclass(frozen=True)
class ActivationDecision:
    desired_plan: PlanId
    recommended_plan: PlanId
    execution_preference: ExecutionPreference
    reasons: tuple[str, ...]
    policy_version: str = ACTIVATION_POLICY_VERSION


def recommend_activation(
    profile: TaskProfile,
    plans: list[Plan] | tuple[Plan, ...],
) -> ActivationDecision:
    """Recommend reasoning depth and execution surface without executing either."""
    plans_by_id = {plan.id: plan for plan in plans}
    required_ids = {"fast", "balanced", "quality"}
    if len(plans) != 3 or set(plans_by_id) != required_ids:
        raise ValueError(
            "activation policy requires exactly fast, balanced, and quality plans"
        )

    reasons: list[str] = []
    desired_plan: PlanId
    if not profile.benefits_from_diversity:
        desired_plan = "fast"
        reasons.append("diversity_not_needed")
    else:
        reasons.append("diversity_beneficial")
        if profile.risk >= 4:
            reasons.append("high_risk")
        if profile.complexity >= 4:
            reasons.append("high_complexity")
        desired_plan = (
            "quality"
            if profile.risk >= 4 or profile.complexity >= 4
            else "balanced"
        )

    execution_preference: ExecutionPreference = "custom_tool_free_ok"
    if profile.needs_tools or profile.needs_freshness:
        execution_preference = "hermes_native_preferred"
    if profile.needs_tools:
        reasons.append("tools_need_native_execution")
    if profile.needs_freshness:
        reasons.append("freshness_need_native_execution")

    candidate_orders: dict[PlanId, tuple[PlanId, ...]] = {
        "fast": ("fast",),
        "balanced": ("balanced", "fast"),
        "quality": ("quality", "balanced", "fast"),
    }
    recommended_plan = desired_plan
    if plans_by_id[desired_plan].degraded:
        reasons.append("desired_plan_degraded")
        recommended_plan = "fast"
        for candidate_id in candidate_orders[desired_plan]:
            if not plans_by_id[candidate_id].degraded:
                recommended_plan = candidate_id
                break
        if recommended_plan != desired_plan:
            reasons.append("recommendation_fallback")
        if all(plan.degraded for plan in plans_by_id.values()):
            reasons.append("all_candidate_plans_degraded")

    return ActivationDecision(
        desired_plan=desired_plan,
        recommended_plan=recommended_plan,
        execution_preference=execution_preference,
        reasons=tuple(reasons),
    )
