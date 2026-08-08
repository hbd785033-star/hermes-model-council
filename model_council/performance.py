"""Offline, non-routing performance comparison for Council plans."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .telemetry import RunPerformanceSummary


@dataclass(frozen=True)
class PlanPerformance:
    plan_id: str
    sample_count: int
    known_outcomes: int
    eligible: bool
    success_rate: float | None
    success_ci_low: float | None
    success_ci_high: float | None
    positive_feedback_rate: float | None
    mean_score: float | None
    mean_latency_ms: float | None
    mean_execution_calls: float | None
    mean_total_tokens: float | None
    success_regret: float | None = None
    score_regret: float | None = None
    latency_regret_ms: float | None = None


@dataclass(frozen=True)
class PerformanceReport:
    task_kind: str
    baseline_plan: str
    minimum_samples: int
    ready: bool
    plans: tuple[PlanPerformance, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_kind": self.task_kind,
            "baseline_plan": self.baseline_plan,
            "minimum_samples": self.minimum_samples,
            "ready": self.ready,
            "plans": [asdict(plan) for plan in self.plans],
            "warnings": list(self.warnings),
        }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one known outcome")
    rate = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z2 / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_performance_report(
    summaries: tuple[RunPerformanceSummary, ...],
    *,
    task_kind: str,
    baseline_plan: str = "fast",
    minimum_samples: int = 30,
) -> PerformanceReport:
    """Compare observed plan metrics without selecting or changing a route."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least 1")
    filtered = tuple(summary for summary in summaries if summary.task_kind == task_kind)
    seen: set[str] = set()
    for summary in filtered:
        if summary.plan_id in seen:
            raise ValueError(f"duplicate performance summary for plan: {summary.plan_id}")
        seen.add(summary.plan_id)

    plans: list[PlanPerformance] = []
    for summary in sorted(filtered, key=lambda item: item.plan_id):
        known_outcomes = int(summary.successes + summary.failures)
        success_rate = (
            summary.successes / known_outcomes if known_outcomes > 0 else None
        )
        ci_low: float | None
        ci_high: float | None
        if known_outcomes > 0:
            ci_low, ci_high = _wilson_interval(summary.successes, known_outcomes)
        else:
            ci_low = ci_high = None
        known_feedback = summary.positive_feedback + summary.negative_feedback
        positive_feedback_rate = (
            summary.positive_feedback / known_feedback if known_feedback > 0 else None
        )
        eligible_plan = known_outcomes >= minimum_samples
        plans.append(
            PlanPerformance(
                plan_id=summary.plan_id,
                sample_count=summary.sample_count,
                known_outcomes=known_outcomes,
                eligible=eligible_plan,
                success_rate=success_rate,
                success_ci_low=ci_low,
                success_ci_high=ci_high,
                positive_feedback_rate=positive_feedback_rate,
                mean_score=summary.mean_score,
                mean_latency_ms=summary.mean_latency_ms,
                mean_execution_calls=summary.mean_execution_calls,
                mean_total_tokens=summary.mean_total_tokens,
            )
        )

    warnings: list[str] = []
    baseline = next((plan for plan in plans if plan.plan_id == baseline_plan), None)
    if baseline is None:
        warnings.append(f"baseline plan '{baseline_plan}' has no run outcomes")
    elif not baseline.eligible:
        warnings.append(
            f"baseline plan '{baseline_plan}' has fewer than {minimum_samples} known outcomes"
        )
    eligible_plans = [
        plan for plan in plans if plan.eligible and plan.success_rate is not None
    ]
    if len(eligible_plans) < 2:
        warnings.append("fewer than two plans meet the minimum known-outcome threshold")
    ready = baseline is not None and baseline.eligible and len(eligible_plans) >= 2

    if ready:
        best_success = max(
            plan.success_rate
            for plan in eligible_plans
            if plan.success_rate is not None
        )
        scored = [
            plan.mean_score for plan in eligible_plans if plan.mean_score is not None
        ]
        latencies = [
            plan.mean_latency_ms
            for plan in eligible_plans
            if plan.mean_latency_ms is not None
        ]
        best_score = max(scored) if scored else None
        best_latency = min(latencies) if latencies else None
        plans = [
            PlanPerformance(
                **{
                    **asdict(plan),
                    "success_regret": (
                        best_success - plan.success_rate
                        if plan.eligible and plan.success_rate is not None
                        else None
                    ),
                    "score_regret": (
                        best_score - plan.mean_score
                        if plan.eligible
                        and best_score is not None
                        and plan.mean_score is not None
                        else None
                    ),
                    "latency_regret_ms": (
                        plan.mean_latency_ms - best_latency
                        if plan.eligible
                        and best_latency is not None
                        and plan.mean_latency_ms is not None
                        else None
                    ),
                }
            )
            for plan in plans
        ]

    return PerformanceReport(
        task_kind=task_kind,
        baseline_plan=baseline_plan,
        minimum_samples=minimum_samples,
        ready=ready,
        plans=tuple(plans),
        warnings=tuple(warnings),
    )
