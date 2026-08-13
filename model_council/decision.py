"""Public decision outcome contracts for HMC processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class DecisionStatus(StrEnum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class DecisionProcess(StrEnum):
    SINGLE = "single"
    CUSTOM_MOA = "custom_moa"
    CUSTOM_COUNCIL = "custom_council"
    NATIVE_MOA = "native_moa"


DECISION_POLICY_VERSION = "hmc-decision-v1.0"


@dataclass(frozen=True)
class CouncilResult:
    """Internal evidence from HMC's native decision process.

    Peer review in this evidence is internal HMC process evidence, not an
    external independent review or evaluation verdict.
    """

    plan_id: str
    final: str
    anonymous_answers: tuple[tuple[str, str], ...]
    reviews: tuple[str, ...]
    failures: tuple[str, ...]
    call_count: int
    degraded: bool = False
    degradation_reason: str | None = None
    candidate_count: int = 0
    review_coverage: float = 0.0
    fallback_source: str | None = None
    task_truncated: bool = False
    actual_process: DecisionProcess | None = None


@dataclass(frozen=True)
class DecisionRecord:
    """Claimed and observed HMC decision-process outcome.

    This record does not assert correctness, verification, confidence, or an
    external evaluation verdict.
    """

    status: DecisionStatus
    decision: str | None
    process: DecisionProcess
    preset: str
    models_consulted: tuple[str, ...]
    configured_call_ceiling: int | None
    topology_required_calls: int
    observed_calls: int | None
    fallback_used: bool = False
    fallback_reason: str | None = None
    degraded_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    policy_version: str = DECISION_POLICY_VERSION
    decision_id: str = field(default_factory=lambda: uuid4().hex)
    process_evidence: CouncilResult | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        has_decision = bool(str(self.decision or "").strip())
        if self.status in {DecisionStatus.COMPLETED, DecisionStatus.DEGRADED} and not has_decision:
            raise ValueError(f"{self.status.value} decision record requires a decision")
        if self.status == DecisionStatus.FAILED and self.decision is not None:
            raise ValueError("failed decision record must not include a decision")
        if self.fallback_used != (self.fallback_reason is not None):
            raise ValueError("fallback_used and fallback_reason must agree")

    @property
    def plan_id(self) -> str:
        return self.preset

    @property
    def final(self) -> str:
        return self.decision or ""

    @property
    def anonymous_answers(self) -> tuple[tuple[str, str], ...]:
        return self.process_evidence.anonymous_answers if self.process_evidence else ()

    @property
    def reviews(self) -> tuple[str, ...]:
        return self.process_evidence.reviews if self.process_evidence else ()

    @property
    def failures(self) -> tuple[str, ...]:
        return self.process_evidence.failures if self.process_evidence else self.warnings

    @property
    def call_count(self) -> int | None:
        return self.observed_calls

    @property
    def degraded(self) -> bool:
        return self.status == DecisionStatus.DEGRADED

    @property
    def degradation_reason(self) -> str | None:
        return self.degraded_reasons[0] if self.degraded_reasons else None

    @property
    def candidate_count(self) -> int:
        return self.process_evidence.candidate_count if self.process_evidence else 0

    @property
    def review_coverage(self) -> float:
        return self.process_evidence.review_coverage if self.process_evidence else 0.0

    @property
    def fallback_source(self) -> str | None:
        return self.process_evidence.fallback_source if self.process_evidence else None

    @property
    def task_truncated(self) -> bool:
        return self.process_evidence.task_truncated if self.process_evidence else False
