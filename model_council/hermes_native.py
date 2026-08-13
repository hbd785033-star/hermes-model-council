"""Truthful adaptation of explicit Hermes-native MoA evidence."""

from __future__ import annotations

from typing import Any

from .decision import DecisionProcess, DecisionRecord, DecisionStatus


def _native_topology_required_calls(normalized_preset: dict[str, Any]) -> int:
    references = normalized_preset.get("reference_models")
    aggregator = normalized_preset.get("aggregator")
    if not isinstance(references, (list, tuple)) or not references:
        raise ValueError("normalized native preset needs reference_models")
    if not isinstance(aggregator, dict) or not aggregator:
        raise ValueError("normalized native preset needs aggregator")
    if any(not isinstance(reference, dict) or not reference for reference in references):
        raise ValueError("normalized native preset has malformed reference_models")
    return len(references) + 1


def adapt_native_moa_outcome(
    *,
    preset: str,
    normalized_preset: dict[str, Any],
    decision: str | None,
    models_consulted: tuple[str, ...] = (),
    observed_calls: int | None = None,
    degraded_reasons: tuple[str, ...] = (),
    fallback_reason: str | None = None,
    warnings: tuple[str, ...] = (),
) -> DecisionRecord:
    """Adapt explicit native config and outcome evidence without execution."""
    topology_required_calls = _native_topology_required_calls(normalized_preset)
    normalized_decision = str(decision or "").strip() or None
    if normalized_decision is None:
        status = DecisionStatus.FAILED
    elif degraded_reasons or fallback_reason is not None:
        status = DecisionStatus.DEGRADED
    else:
        status = DecisionStatus.COMPLETED
    return DecisionRecord(
        status=status,
        decision=normalized_decision,
        process=DecisionProcess.NATIVE_MOA,
        preset=preset,
        models_consulted=tuple(dict.fromkeys(models_consulted)),
        configured_call_ceiling=None,
        topology_required_calls=topology_required_calls,
        observed_calls=observed_calls,
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
        degraded_reasons=degraded_reasons,
        warnings=warnings,
    )
