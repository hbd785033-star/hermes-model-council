"""Pure typed planner contract for HMC recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .activation import (
    ExecutionPreference,
    PlanId,
    recommend_activation,
)
from .analysis import TaskProfile
from .lenses import LENS_POLICY_VERSION, resolve_advisor_lens
from .recommender import Plan

HMC_PLANNER_CONTRACT_VERSION = "hmc-planner-v1.0"

PlanMode = Literal["single", "moa", "council"]

_PLAN_IDS = {"fast", "balanced", "quality"}
_PLAN_MODES = {"single", "moa", "council"}
_LENS_IDS = {"solution", "risk", "feasibility"}


@dataclass(frozen=True)
class PlannerRequest:
    """Validated structured input to the pure HMC planner contract."""

    task_profile: TaskProfile
    contract_version: str = HMC_PLANNER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task_profile, TaskProfile):
            raise ValueError("task_profile must be an HMC TaskProfile")
        if self.contract_version != HMC_PLANNER_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must be {HMC_PLANNER_CONTRACT_VERSION!r}"
            )


@dataclass(frozen=True)
class PlannerCandidate:
    """Normalized planning snapshot without participant or model identity."""

    id: PlanId
    mode: PlanMode
    estimated_calls: int
    max_calls: int
    degraded: bool
    degradation_reason: str | None
    planned_lens_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.id not in _PLAN_IDS:
            raise ValueError("candidate id must be fast, balanced, or quality")
        if self.mode not in _PLAN_MODES:
            raise ValueError("candidate mode must be single, moa, or council")
        if type(self.estimated_calls) is not int or self.estimated_calls < 1:
            raise ValueError("estimated_calls must be a positive integer")
        if type(self.max_calls) is not int or self.max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if self.estimated_calls > self.max_calls:
            raise ValueError("estimated_calls must not exceed max_calls")
        if type(self.degraded) is not bool:
            raise ValueError("degraded must be a boolean")
        if self.degradation_reason is not None and not self.degradation_reason.strip():
            raise ValueError("degradation_reason must not be blank")
        if not self.degraded and self.degradation_reason is not None:
            raise ValueError("non-degraded candidate cannot have degradation_reason")
        if (
            not isinstance(self.planned_lens_ids, tuple)
            or len(self.planned_lens_ids) != len(set(self.planned_lens_ids))
            or any(lens_id not in _LENS_IDS for lens_id in self.planned_lens_ids)
        ):
            raise ValueError("planned_lens_ids must be unique known lens IDs")


@dataclass(frozen=True)
class PlannerRecommendation:
    """Immutable planning recommendation without runtime or execution evidence."""

    planner_contract_version: str
    desired_plan: PlanId
    recommended_plan: PlanId
    execution_preference: ExecutionPreference
    activation_reasons: tuple[str, ...]
    activation_policy_version: str
    selected_plan_mode: PlanMode
    degraded: bool
    degradation_reason: str | None
    planned_call_count: int
    planner_call_ceiling: int
    planned_lens_ids: tuple[str, ...]
    lens_policy_version: str


def _normalize_plan(plan: Plan) -> PlannerCandidate:
    if not isinstance(plan, Plan):
        raise ValueError("planner candidates must be HMC Plan instances")
    if plan.id not in _PLAN_IDS:
        raise ValueError("candidate id must be fast, balanced, or quality")
    if plan.mode not in _PLAN_MODES:
        raise ValueError("candidate mode must be single, moa, or council")

    planned_lens_ids = tuple(
        resolve_advisor_lens(participant.role).id
        for participant in plan.participants
        if participant.role.startswith("advisor")
    )
    return PlannerCandidate(
        id=cast(PlanId, plan.id),
        mode=cast(PlanMode, plan.mode),
        estimated_calls=plan.estimated_calls,
        max_calls=plan.max_calls,
        degraded=plan.degraded,
        degradation_reason=plan.degradation_reason,
        planned_lens_ids=planned_lens_ids,
    )


def plan_task(
    request: PlannerRequest,
    plans: list[Plan] | tuple[Plan, ...],
) -> PlannerRecommendation:
    """Compose a deterministic planning recommendation without execution or I/O."""
    if not isinstance(request, PlannerRequest):
        raise ValueError("request must be a PlannerRequest")

    if not all(isinstance(plan, Plan) for plan in plans):
        raise ValueError("planner candidates must be HMC Plan instances")
    plan_ids = [plan.id for plan in plans]
    if len(plans) != 3 or len(set(plan_ids)) != 3 or set(plan_ids) != _PLAN_IDS:
        raise ValueError(
            "planner contract requires exactly fast, balanced, and quality candidates"
        )

    candidates = tuple(_normalize_plan(plan) for plan in plans)
    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    activation = recommend_activation(request.task_profile, plans)
    selected = candidates_by_id.get(activation.recommended_plan)
    if selected is None:
        raise ValueError("recommended candidate is missing")

    return PlannerRecommendation(
        planner_contract_version=request.contract_version,
        desired_plan=activation.desired_plan,
        recommended_plan=activation.recommended_plan,
        execution_preference=activation.execution_preference,
        activation_reasons=activation.reasons,
        activation_policy_version=activation.policy_version,
        selected_plan_mode=selected.mode,
        degraded=selected.degraded,
        degradation_reason=selected.degradation_reason,
        planned_call_count=selected.estimated_calls,
        planner_call_ceiling=selected.max_calls,
        planned_lens_ids=selected.planned_lens_ids,
        lens_policy_version=LENS_POLICY_VERSION,
    )
