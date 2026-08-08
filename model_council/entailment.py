"""Advisory claim-citation entailment policy.

This module aggregates results from trusted evaluators; it does not implement an
LLM judge. Evaluator outputs are untrusted until their evaluator ID is explicitly
allowed and its calibration record meets the hard-gate policy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .evidence import EvidenceBundle


class EntailmentVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class EntailmentAssessment:
    id: str
    claim_id: str
    evidence_id: str
    verdict: EntailmentVerdict
    evaluator: str
    rationale: str = ""


@dataclass(frozen=True)
class EvaluatorCalibration:
    evaluator: str
    dataset_version: str
    sample_count: int
    pearson_correlation: float
    severe_false_accept_rate: float

    def __post_init__(self) -> None:
        if not self.evaluator.strip():
            raise ValueError("calibration evaluator must not be empty")
        if not self.dataset_version.strip():
            raise ValueError("calibration dataset version must not be empty")
        if self.sample_count < 0:
            raise ValueError("calibration sample count must not be negative")
        if not -1.0 <= self.pearson_correlation <= 1.0:
            raise ValueError("Pearson correlation must be between -1 and 1")
        if not 0.0 <= self.severe_false_accept_rate <= 1.0:
            raise ValueError("false-accept rate must be between 0 and 1")


@dataclass(frozen=True)
class EntailmentClaimResult:
    claim_id: str
    verdict: EntailmentVerdict
    evaluator_ids: tuple[str, ...]
    disagreement: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict.value,
            "evaluator_ids": list(self.evaluator_ids),
            "disagreement": self.disagreement,
        }


@dataclass(frozen=True)
class EntailmentPolicyResult:
    claims: tuple[EntailmentClaimResult, ...]
    advisory: bool
    hard_gate_eligible: bool
    should_block: bool
    untrusted_assessment_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "advisory": self.advisory,
            "hard_gate_eligible": self.hard_gate_eligible,
            "should_block": self.should_block,
            "untrusted_assessment_ids": list(self.untrusted_assessment_ids),
        }


class EntailmentPolicy:
    """Aggregate trusted entailment judgments without pretending they are truth."""

    def __init__(
        self,
        *,
        trusted_evaluators: tuple[str, ...],
        enable_hard_gate: bool = False,
        calibrations: tuple[EvaluatorCalibration, ...] = (),
        minimum_samples: int = 100,
        minimum_pearson: float = 0.7,
        maximum_severe_false_accept_rate: float = 0.05,
    ) -> None:
        self.trusted_evaluators = {
            evaluator.strip()
            for evaluator in trusted_evaluators
            if evaluator.strip()
        }
        self.enable_hard_gate = bool(enable_hard_gate)
        self.calibrations = {calibration.evaluator: calibration for calibration in calibrations}
        self.minimum_samples = max(1, int(minimum_samples))
        self.minimum_pearson = float(minimum_pearson)
        self.maximum_severe_false_accept_rate = float(maximum_severe_false_accept_rate)
        if not -1.0 <= self.minimum_pearson <= 1.0:
            raise ValueError("minimum Pearson correlation must be between -1 and 1")
        if not 0.0 <= self.maximum_severe_false_accept_rate <= 1.0:
            raise ValueError("maximum false-accept rate must be between 0 and 1")

    def _calibrated(self, evaluators: set[str]) -> bool:
        if not evaluators:
            return False
        for evaluator in evaluators:
            calibration = self.calibrations.get(evaluator)
            if calibration is None:
                return False
            if calibration.sample_count < self.minimum_samples:
                return False
            if calibration.pearson_correlation < self.minimum_pearson:
                return False
            if calibration.severe_false_accept_rate > self.maximum_severe_false_accept_rate:
                return False
        return True

    @staticmethod
    def _claim_verdict(verdicts: set[EntailmentVerdict]) -> tuple[EntailmentVerdict, bool]:
        if EntailmentVerdict.SUPPORTED in verdicts and EntailmentVerdict.CONTRADICTED in verdicts:
            return EntailmentVerdict.INSUFFICIENT, True
        if EntailmentVerdict.CONTRADICTED in verdicts:
            return EntailmentVerdict.CONTRADICTED, False
        if EntailmentVerdict.SUPPORTED in verdicts:
            return EntailmentVerdict.SUPPORTED, False
        return EntailmentVerdict.INSUFFICIENT, False

    def evaluate(
        self,
        bundle: EvidenceBundle,
        assessments: tuple[EntailmentAssessment, ...],
    ) -> EntailmentPolicyResult:
        claim_ids = {claim.id for claim in bundle.claims}
        evidence_ids = {artifact.id for artifact in bundle.artifacts}
        assessment_ids: set[str] = set()
        trusted_by_claim: dict[str, list[EntailmentAssessment]] = defaultdict(list)
        untrusted_ids: list[str] = []
        for assessment in assessments:
            if assessment.evaluator not in self.trusted_evaluators:
                untrusted_ids.append(assessment.id or "[missing-id]")
                continue
            if not assessment.id.strip():
                raise ValueError("entailment assessment ID must not be empty")
            if assessment.id in assessment_ids:
                raise ValueError(f"duplicate entailment assessment ID: {assessment.id}")
            assessment_ids.add(assessment.id)
            if assessment.claim_id not in claim_ids:
                raise ValueError(f"assessment references unknown claim: {assessment.claim_id}")
            if assessment.evidence_id not in evidence_ids:
                raise ValueError(f"assessment references unknown evidence: {assessment.evidence_id}")
            trusted_by_claim[assessment.claim_id].append(assessment)

        claim_results: list[EntailmentClaimResult] = []
        used_evaluators: set[str] = set()
        required_failures = False
        for claim in bundle.claims:
            trusted = trusted_by_claim.get(claim.id, [])
            verdict, disagreement = self._claim_verdict({item.verdict for item in trusted})
            evaluator_ids = tuple(sorted({item.evaluator for item in trusted}))
            used_evaluators.update(evaluator_ids)
            claim_results.append(
                EntailmentClaimResult(claim.id, verdict, evaluator_ids, disagreement)
            )
            if claim.importance.value == "required" and verdict != EntailmentVerdict.SUPPORTED:
                required_failures = True

        hard_gate_eligible = self.enable_hard_gate and self._calibrated(used_evaluators)
        return EntailmentPolicyResult(
            claims=tuple(claim_results),
            advisory=not hard_gate_eligible,
            hard_gate_eligible=hard_gate_eligible,
            should_block=hard_gate_eligible and required_failures,
            untrusted_assessment_ids=tuple(untrusted_ids),
        )
