"""Advisory-only shadow routing proposals from offline performance evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .performance import PerformanceReport


@dataclass(frozen=True)
class ShadowRouteProposal:
    task_kind: str
    baseline_plan: str
    candidate_plan: str
    baseline_samples: int
    candidate_samples: int
    baseline_success_ci_high: float
    candidate_success_ci_low: float
    score_delta: float
    latency_delta_ms: float | None
    reason: str
    rollback_conditions: tuple[str, ...]


@dataclass(frozen=True)
class ShadowRouterReport:
    task_kind: str
    baseline_plan: str
    ready: bool
    apply_automatically: bool
    proposals: tuple[ShadowRouteProposal, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_kind": self.task_kind,
            "baseline_plan": self.baseline_plan,
            "ready": self.ready,
            "apply_automatically": False,
            "proposals": [asdict(proposal) for proposal in self.proposals],
            "warnings": list(self.warnings),
        }


def build_shadow_router_report(performance: PerformanceReport) -> ShadowRouterReport:
    """Propose shadow candidates only for statistically separated quality gains."""
    warnings = list(performance.warnings)
    if not performance.ready:
        warnings.append("performance report is not ready; no shadow route can be proposed")
        return ShadowRouterReport(
            task_kind=performance.task_kind,
            baseline_plan=performance.baseline_plan,
            ready=False,
            apply_automatically=False,
            proposals=(),
            warnings=tuple(warnings),
        )

    baseline = next(
        (plan for plan in performance.plans if plan.plan_id == performance.baseline_plan),
        None,
    )
    if baseline is None or baseline.success_ci_high is None:
        warnings.append("baseline confidence interval is unavailable")
        return ShadowRouterReport(
            task_kind=performance.task_kind,
            baseline_plan=performance.baseline_plan,
            ready=False,
            apply_automatically=False,
            proposals=(),
            warnings=tuple(warnings),
        )

    proposals: list[ShadowRouteProposal] = []
    for candidate in performance.plans:
        if candidate.plan_id == baseline.plan_id or not candidate.eligible:
            continue
        if candidate.success_ci_low is None:
            warnings.append(f"plan '{candidate.plan_id}' has no success confidence interval")
            continue
        if candidate.success_ci_low <= baseline.success_ci_high:
            warnings.append(
                f"plan '{candidate.plan_id}' success confidence interval overlaps baseline"
            )
            continue
        if baseline.mean_score is None or candidate.mean_score is None:
            warnings.append(
                f"plan '{candidate.plan_id}' lacks external score evidence for shadow routing"
            )
            continue
        score_delta = round(candidate.mean_score - baseline.mean_score, 12)
        if score_delta < 0:
            warnings.append(
                f"plan '{candidate.plan_id}' has a lower external evaluator score"
            )
            continue
        latency_delta = (
            candidate.mean_latency_ms - baseline.mean_latency_ms
            if candidate.mean_latency_ms is not None
            and baseline.mean_latency_ms is not None
            else None
        )
        proposals.append(
            ShadowRouteProposal(
                task_kind=performance.task_kind,
                baseline_plan=baseline.plan_id,
                candidate_plan=candidate.plan_id,
                baseline_samples=baseline.known_outcomes,
                candidate_samples=candidate.known_outcomes,
                baseline_success_ci_high=baseline.success_ci_high,
                candidate_success_ci_low=candidate.success_ci_low,
                score_delta=score_delta,
                latency_delta_ms=latency_delta,
                reason=(
                    "candidate success lower bound exceeds baseline upper bound and "
                    "external evaluator score does not regress"
                ),
                rollback_conditions=(
                    "candidate success interval no longer clears the baseline interval",
                    "candidate external evaluator score falls below the baseline",
                    "candidate severe failure or negative-feedback rate exceeds policy limits",
                ),
            )
        )

    if not proposals:
        warnings.append("no plan satisfies the conservative shadow-routing evidence rule")
    return ShadowRouterReport(
        task_kind=performance.task_kind,
        baseline_plan=performance.baseline_plan,
        ready=True,
        apply_automatically=False,
        proposals=tuple(proposals),
        warnings=tuple(warnings),
    )
