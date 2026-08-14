"""Command-line interface for Hermes Model Council."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .activation import recommend_activation
from .analysis import TaskProfile, analyze_task
from .decision import CouncilResult, DecisionRecord
from .health import HealthCache, ProbeResult, probe_models
from .hermes_config import backup_hermes_config as _backup_hermes_config  # noqa: F401
from .hermes_config import check_hermes_config as _check_hermes_config  # noqa: F401
from .hermes_config import install_native_presets as _install_presets
from .hermes_config import (
    save_config_with_rollback as _save_config_with_rollback,  # noqa: F401
)
from .hermes_invoker import HermesInvoker
from .inventory import ModelSpec, discover_models
from .recommender import Plan, recommend_plans
from .runner import CouncilRunner


def _health_cache_path() -> Path:
    configured_dir = os.environ.get("MODEL_COUNCIL_CACHE_DIR")
    cache_dir = (
        Path(configured_dir)
        if configured_dir
        else Path.home() / ".cache" / "hermes-model-council"
    )
    return cache_dir / "health-cache.json"


_HEALTH_CACHE_PATH = _health_cache_path()
_HEALTH_CACHE_TTL_SECONDS = 900


def _store_health_cache(cache: HealthCache, health: dict[str, bool]) -> bool:
    try:
        cache.store(health)
    except OSError as exc:
        print(
            f"Health cache write skipped: {type(exc).__name__}",
            file=sys.stderr,
        )
        return False
    return True


def model_to_dict(model: ModelSpec) -> dict[str, Any]:
    return asdict(model)


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "label": plan.label,
        "mode": plan.mode,
        "estimated_calls": plan.estimated_calls,
        "max_calls": plan.max_calls,
        "degraded": plan.degraded,
        "degradation_reason": plan.degradation_reason,
        "strengths": list(plan.strengths),
        "risks": list(plan.risks),
        "participants": [
            {
                "role": participant.role,
                "provider": participant.model.provider,
                "model": participant.model.model,
                "family": participant.model.family,
                "healthy": participant.model.healthy,
                "reasoning_effort": participant.reasoning_effort,
            }
            for participant in plan.participants
        ],
    }


def probe_candidates(
    plans: list[Plan], *, exclude_keys: set[str] | None = None
) -> list[ModelSpec]:
    found: list[ModelSpec] = []
    seen: set[str] = set(exclude_keys or ())
    for plan in plans:
        for participant in plan.participants:
            model = participant.model
            if model.key not in seen:
                seen.add(model.key)
                found.append(model)
    return found


def _merge_health(
    models: list[ModelSpec], result: ProbeResult
) -> list[ModelSpec]:
    exact = {model.key: model.healthy for model in result.models}
    probed_providers = {model.provider for model in result.models}
    provider_success: dict[str, bool] = {}
    for model in result.models:
        provider_success[model.provider] = (
            provider_success.get(model.provider, False) or model.healthy is True
        )
    unavailable_providers = {
        provider
        for provider in probed_providers
        if not provider_success.get(provider, False)
    }
    updated: list[ModelSpec] = []
    for model in models:
        if model.key in exact:
            updated.append(replace(model, healthy=exact[model.key]))
        elif model.healthy is True:
            updated.append(model)
        elif model.provider in unavailable_providers:
            updated.append(replace(model, healthy=False))
        else:
            updated.append(model)
    return updated


def _only_verified(models: list[ModelSpec]) -> list[ModelSpec]:
    """After a live probe, unknown is not treated as available."""
    return [
        model if model.healthy is not None else replace(model, healthy=False)
        for model in models
    ]


def prepare_recommendation(
    task: str,
    *,
    live_probe: bool,
    timeout: int,
    refresh_probe: bool = False,
) -> tuple[TaskProfile, list[ModelSpec], list[Plan], dict[str, str], int, int]:
    profile = analyze_task(task)
    models = discover_models()
    plans = recommend_plans(profile, models)
    diagnostics: dict[str, str] = {}
    probe_call_count = 0
    probe_cache_hit_count = 0
    if live_probe:
        invoker = HermesInvoker(timeout=timeout)
        cache = HealthCache(_HEALTH_CACHE_PATH, _HEALTH_CACHE_TTL_SECONDS)
        cached = {} if refresh_probe else cache.load(models)
        probe_cache_hit_count = len(cached)
        if cached:
            cached_models = tuple(
                replace(model, healthy=cached[model.key])
                for model in models
                if model.key in cached
            )
            cached_result = ProbeResult(
                cached_models,
                {
                    model.key: "cached: ok" if model.healthy else "cached: unavailable"
                    for model in cached_models
                },
            )
            diagnostics.update(cached_result.diagnostics)
            models = _merge_health(models, cached_result)
            plans = recommend_plans(profile, models)
        checked: set[str] = set(cached)
        for _ in range(2):
            candidates = probe_candidates(plans, exclude_keys=checked)
            if not candidates:
                break
            result = probe_models(candidates, invoke=invoker, max_workers=3)
            probe_call_count += len(result.models)
            checked.update(model.key for model in result.models)
            diagnostics.update(result.diagnostics)
            _store_health_cache(
                cache,
                {model.key: bool(model.healthy) for model in result.models},
            )
            models = _merge_health(models, result)
            plans = recommend_plans(profile, models)
        models = _only_verified(models)
        plans = recommend_plans(profile, models)
    return (
        profile,
        models,
        plans,
        diagnostics,
        probe_call_count,
        probe_cache_hit_count,
    )


def _print_inventory(models: list[ModelSpec], as_json: bool) -> None:
    if as_json:
        print(json.dumps([model_to_dict(model) for model in models], ensure_ascii=False, indent=2))
        return
    by_provider: dict[str, list[ModelSpec]] = {}
    for model in models:
        by_provider.setdefault(model.provider, []).append(model)
    print(f"Discovered {len(models)} selectable models across {len(by_provider)} providers")
    for provider, rows in by_provider.items():
        print(f"\n{provider} ({len(rows)})")
        for model in rows:
            marker = " *current" if model.is_current else ""
            print(f"  - {model.model}{marker}")


def _recommendation_payload(
    profile: TaskProfile,
    models: list[ModelSpec],
    plans: list[Plan],
    diagnostics: dict[str, str],
    probe_call_count: int = 0,
    probe_cache_hit_count: int = 0,
) -> dict[str, Any]:
    activation = recommend_activation(profile, plans)
    return {
        "task_profile": asdict(profile),
        "available_models": [model_to_dict(model) for model in models if model.healthy is not False],
        "plans": [plan_to_dict(plan) for plan in plans],
        "health_diagnostics": diagnostics,
        "probe_call_count": probe_call_count,
        "probe_cache_hit_count": probe_cache_hit_count,
        "activation": {
            "desired_plan": activation.desired_plan,
            "recommended_plan": activation.recommended_plan,
            "execution_preference": activation.execution_preference,
            "reasons": list(activation.reasons),
            "policy_version": activation.policy_version,
        },
    }


def _print_recommendation(
    profile: TaskProfile,
    models: list[ModelSpec],
    plans: list[Plan],
    diagnostics: dict[str, str],
    probe_call_count: int,
    probe_cache_hit_count: int,
    as_json: bool,
) -> None:
    payload = _recommendation_payload(
        profile,
        models,
        plans,
        diagnostics,
        probe_call_count,
        probe_cache_hit_count,
    )
    if as_json:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        f"Task: type={profile.kind}, complexity={profile.complexity}/5, "
        f"risk={profile.risk}/5, tools={'yes' if profile.needs_tools else 'no'}"
    )
    activation = payload["activation"]
    print(f"Desired plan: {activation['desired_plan']}")
    print(f"Recommended plan: {activation['recommended_plan']}")
    print(f"Execution preference: {activation['execution_preference']}")
    print("Activation reasons: " + ", ".join(activation["reasons"]))
    print(f"Activation policy: {activation['policy_version']}")
    if diagnostics:
        print("\nHealth probes:")
        for key, status in diagnostics.items():
            print(f"  - {key}: {status}")
        print(f"  actual calls: {probe_call_count}; cache hits: {probe_cache_hit_count}")
    for plan in plans:
        print(f"\n[{plan.id}] {plan.label} — mode={plan.mode}, calls≈{plan.estimated_calls}")
        for participant in plan.participants:
            print(
                f"  - {participant.role}: {participant.model.key} "
                f"[reasoning={participant.reasoning_effort}]"
            )
        print(f"  strengths: {', '.join(plan.strengths)}")
        print(f"  risks: {', '.join(plan.risks)}")


def _result_payload(
    result: DecisionRecord | CouncilResult,
    *,
    probe_call_count: int = 0,
    probe_cache_hit_count: int = 0,
) -> dict[str, Any]:
    process: str | None
    if isinstance(result, DecisionRecord):
        status = result.status.value
        decision = result.decision
        decision_id = result.decision_id
        process = result.process.value
        preset = result.preset
        policy_version = result.policy_version
        models_consulted = list(result.models_consulted)
        configured_call_ceiling = result.configured_call_ceiling
        topology_required_calls = result.topology_required_calls
        observed_calls = result.observed_calls
        fallback_used = result.fallback_used
        fallback_reason = result.fallback_reason
        degraded_reasons = list(result.degraded_reasons)
        warnings = list(result.warnings)
    else:
        status = "degraded" if result.degraded else "completed"
        decision = result.final or None
        decision_id = None
        process = result.actual_process.value if result.actual_process is not None else None
        preset = result.plan_id
        policy_version = None
        models_consulted = []
        configured_call_ceiling = None
        topology_required_calls = None
        observed_calls = result.call_count
        fallback_used = result.fallback_source is not None
        fallback_reason = result.degradation_reason if fallback_used else None
        degraded_reasons = (
            [result.degradation_reason]
            if result.degradation_reason is not None
            else []
        )
        warnings = list(result.failures)
    total = probe_call_count + observed_calls if observed_calls is not None else None
    return {
        "plan": result.plan_id,
        "final": result.final,
        "status": status,
        "decision": decision,
        "decision_id": decision_id,
        "process": process,
        "preset": preset,
        "policy_version": policy_version,
        "models_consulted": models_consulted,
        "configured_call_ceiling": configured_call_ceiling,
        "topology_required_calls": topology_required_calls,
        "observed_calls": observed_calls,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "degraded_reasons": degraded_reasons,
        "warnings": warnings,
        "anonymous_answers": [
            {"label": label, "text": text} for label, text in result.anonymous_answers
        ],
        "reviews": list(result.reviews),
        "failures": list(result.failures),
        "call_count": result.call_count,
        "probe_call_count": probe_call_count,
        "probe_cache_hit_count": probe_cache_hit_count,
        "execution_call_count": result.call_count,
        "total_call_count": total,
        "degraded": result.degraded,
        "degradation_reason": result.degradation_reason,
        "candidate_count": result.candidate_count,
        "review_coverage": result.review_coverage,
        "fallback_source": result.fallback_source,
        "task_truncated": result.task_truncated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-council",
        description="Task-aware model plans and anonymous multi-model councils for Hermes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="List configured Hermes models")
    inventory.add_argument("--json", action="store_true")

    recommend = sub.add_parser("recommend", help="Recommend fast, balanced and quality plans")
    recommend.add_argument("task")
    recommend.add_argument("--probe", action="store_true", help="Run live health checks")
    recommend.add_argument("--refresh-probe", action="store_true", help="Ignore cached health results")
    recommend.add_argument("--timeout", type=int, default=180)
    recommend.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="Execute one recommended plan")
    run.add_argument("task")
    run.add_argument("--plan", choices=("fast", "balanced", "quality"), required=True)
    run.add_argument("--no-probe", action="store_true")
    run.add_argument("--refresh-probe", action="store_true", help="Ignore cached health results")
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--yes", action="store_true", help="Confirm the displayed model-call budget")
    run.add_argument("--json", action="store_true")

    install = sub.add_parser("install-presets", help="Install native Hermes MoA presets")
    install.add_argument(
        "--task",
        default="复杂、高风险的通用任务，需要独立分析、工具执行和最终审查",
    )
    install.add_argument("--no-probe", action="store_true")
    install.add_argument("--refresh-probe", action="store_true", help="Ignore cached health results")
    install.add_argument("--timeout", type=int, default=180)
    install.add_argument("--yes", action="store_true", help="Confirm config backup and write")
    install.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        _print_inventory(discover_models(), args.json)
        return 0

    if args.command == "recommend":
        (
            profile,
            models,
            plans,
            diagnostics,
            probe_call_count,
            probe_cache_hit_count,
        ) = prepare_recommendation(
            args.task,
            live_probe=args.probe,
            timeout=args.timeout,
            refresh_probe=args.refresh_probe,
        )
        _print_recommendation(
            profile,
            models,
            plans,
            diagnostics,
            probe_call_count,
            probe_cache_hit_count,
            args.json,
        )
        return 0

    if args.command == "run":
        if not args.yes:
            raise SystemExit("Refusing to execute model calls without --yes")
        (
            profile,
            models,
            plans,
            diagnostics,
            probe_call_count,
            probe_cache_hit_count,
        ) = prepare_recommendation(
            args.task,
            live_probe=not args.no_probe,
            timeout=args.timeout,
            refresh_probe=args.refresh_probe,
        )
        plan = next(plan for plan in plans if plan.id == args.plan)
        runner = CouncilRunner(HermesInvoker(timeout=args.timeout), max_workers=3)
        result = runner.run(args.task, plan)
        payload = _result_payload(
            result,
            probe_call_count=probe_call_count,
            probe_cache_hit_count=probe_cache_hit_count,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(result.final)
            if result.failures:
                print("\nDegraded participants:", file=sys.stderr)
                for failure in result.failures:
                    print(f"- {failure}", file=sys.stderr)
            print(
                "\n[model-council calls: "
                f"probe={probe_call_count}, execution={result.call_count}, "
                f"total={payload['total_call_count']}, cache_hits={probe_cache_hit_count}]",
                file=sys.stderr,
            )
        return 0

    if args.command == "install-presets":
        if not args.yes:
            raise SystemExit("Refusing to modify Hermes config without --yes")
        (
            profile,
            models,
            plans,
            diagnostics,
            probe_call_count,
            probe_cache_hit_count,
        ) = prepare_recommendation(
            args.task,
            live_probe=not args.no_probe,
            timeout=args.timeout,
            refresh_probe=args.refresh_probe,
        )
        backup, moa = _install_presets(plans)
        preset_names = list((moa.get("presets") or {}).keys())
        payload = {
            "backup": str(backup),
            "default_preset": moa.get("default_preset"),
            "presets": preset_names,
            "health_diagnostics": diagnostics,
            "probe_call_count": probe_call_count,
            "probe_cache_hit_count": probe_cache_hit_count,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Installed native Hermes MoA presets. Backup: {backup}")
            print(f"Default preset: {payload['default_preset']}")
            print("Presets: " + ", ".join(str(name) for name in preset_names))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
