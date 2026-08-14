"""Pure deterministic decision-lens policy for custom HMC advisors."""

from __future__ import annotations

from dataclasses import dataclass

LENS_POLICY_VERSION = "hmc-lenses-v1.0"


@dataclass(frozen=True)
class DecisionLens:
    id: str
    instruction: str


SOLUTION_LENS = DecisionLens(
    "solution",
    "Construct the strongest direct solution: state a clear position, identify key "
    "assumptions, present the strongest supporting reasoning or evidence, and give a "
    "concrete recommendation.",
)
RISK_LENS = DecisionLens(
    "risk",
    "Actively stress-test the proposed direction: seek counterexamples, hidden "
    "assumptions, failure modes, correctness risks, security or safety risks when "
    "applicable, and conditions that would reverse the recommendation.",
)
FEASIBILITY_LENS = DecisionLens(
    "feasibility",
    "Test whether the proposal can be implemented and operated: examine implementation "
    "constraints, dependencies, operational complexity, cost and latency, maintainability, "
    "reversibility, rollout, and a measurable next action.",
)

DECISION_LENSES = (SOLUTION_LENS, RISK_LENS, FEASIBILITY_LENS)

_ROLE_LENSES = {
    "advisor": SOLUTION_LENS,
    "advisor-1": SOLUTION_LENS,
    "advisor-2": RISK_LENS,
    "advisor-3": FEASIBILITY_LENS,
    "advisor-solution": SOLUTION_LENS,
    "advisor-risk": RISK_LENS,
    "advisor-feasibility": FEASIBILITY_LENS,
}


def select_decision_lenses(advisor_count: int) -> tuple[DecisionLens, ...]:
    """Return the exact V1 lens prefix for an advisor count from zero to three."""
    if type(advisor_count) is not int:
        raise ValueError("advisor_count must be an integer between 0 and 3")
    if advisor_count < 0 or advisor_count > len(DECISION_LENSES):
        raise ValueError("advisor_count must be an integer between 0 and 3")
    return DECISION_LENSES[:advisor_count]


def resolve_advisor_lens(role: str) -> DecisionLens:
    """Resolve explicit and legacy advisor roles with a fail-soft advisor fallback."""
    if role in _ROLE_LENSES:
        return _ROLE_LENSES[role]
    if role.startswith("advisor"):
        return SOLUTION_LENS
    raise ValueError(f"role is not an advisor: {role}")
