"""Create native Hermes MoA presets from recommended plans."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .decision import DecisionRecord
from .hermes_native import adapt_native_moa_outcome
from .recommender import Participant, Plan

BALANCED_PRESET = "model-council-balanced"
QUALITY_PRESET = "model-council-quality"


def _slot(participant: Participant, *, enabled: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "provider": participant.model.provider,
        "model": participant.model.model,
        "reasoning_effort": participant.reasoning_effort,
    }
    if enabled:
        value["enabled"] = True
    return value


def _preset(plan: Plan, *, reference_max_tokens: int) -> dict[str, Any]:
    aggregator = plan.chairman
    references = [
        participant
        for participant in plan.participants
        if participant.role.startswith("advisor")
        and participant.model.key != aggregator.model.key
    ]
    if not references:
        references = [
            participant
            for participant in plan.participants
            if participant.role.startswith("advisor")
        ]
    if not references:
        references = [aggregator]
    return {
        "enabled": True,
        "reference_models": [_slot(participant, enabled=True) for participant in references],
        "aggregator": _slot(aggregator),
        "reference_temperature": None,
        "aggregator_temperature": None,
        "reference_timeout": None,
        "degraded_reference_policy": "loud",
        "max_tokens": 4096,
        "reference_max_tokens": reference_max_tokens,
        "fanout": "user_turn",
    }


def build_native_moa_config(
    plans: list[Plan], existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Add safe named presets while preserving unrelated user presets."""
    by_id = {plan.id: plan for plan in plans}
    if "balanced" not in by_id or "quality" not in by_id:
        raise ValueError("balanced and quality plans are required")
    raw = deepcopy(existing) if isinstance(existing, dict) else {}
    stored_presets = raw.get("presets")
    presets: dict[str, Any] = (
        deepcopy(stored_presets) if isinstance(stored_presets, dict) else {}
    )
    presets[BALANCED_PRESET] = _preset(by_id["balanced"], reference_max_tokens=600)
    presets[QUALITY_PRESET] = _preset(by_id["quality"], reference_max_tokens=900)
    active = str(raw.get("active_preset") or "").strip()
    if active not in presets:
        active = ""
    result: dict[str, Any] = {
        **raw,
        "default_preset": BALANCED_PRESET,
        "active_preset": active,
        "presets": presets,
        "privacy_filter": "full",
    }
    return result


def native_moa_decision_record(
    plan: Plan,
    *,
    normalized_preset: dict[str, Any] | None = None,
    decision: str | None,
    models_consulted: tuple[str, ...] = (),
    observed_calls: int | None = None,
    degraded_reasons: tuple[str, ...] = (),
    fallback_reason: str | None = None,
    warnings: tuple[str, ...] = (),
) -> DecisionRecord:
    """Record an externally executed native Hermes MoA decision truthfully.

    Building this record does not execute Hermes or claim external evaluation.
    Native MoA uses reference models followed by an aggregator; it is not the
    custom anonymous-review Council process even when the preset is `quality`.
    """
    if normalized_preset is None:
        raise ValueError("normalized native preset evidence is required")
    return adapt_native_moa_outcome(
        preset=plan.id,
        normalized_preset=normalized_preset,
        decision=decision,
        models_consulted=models_consulted,
        observed_calls=observed_calls,
        degraded_reasons=degraded_reasons,
        fallback_reason=fallback_reason,
        warnings=warnings,
    )
