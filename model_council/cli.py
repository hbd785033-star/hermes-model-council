"""Command-line interface for Hermes Model Council."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .analysis import TaskProfile, analyze_task
from .health import ProbeResult, probe_models
from .hermes_invoker import HermesInvoker
from .inventory import ModelSpec, discover_models
from .presets import build_native_moa_config
from .recommender import Plan, recommend_plans
from .runner import CouncilResult, CouncilRunner


def model_to_dict(model: ModelSpec) -> dict[str, Any]:
    return asdict(model)


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "label": plan.label,
        "mode": plan.mode,
        "estimated_calls": plan.estimated_calls,
        "max_calls": plan.max_calls,
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
) -> tuple[TaskProfile, list[ModelSpec], list[Plan], dict[str, str]]:
    profile = analyze_task(task)
    models = discover_models()
    plans = recommend_plans(profile, models)
    diagnostics: dict[str, str] = {}
    if live_probe:
        invoker = HermesInvoker(timeout=timeout)
        checked: set[str] = set()
        for _ in range(2):
            candidates = probe_candidates(plans, exclude_keys=checked)
            if not candidates:
                break
            result = probe_models(candidates, invoke=invoker, max_workers=3)
            checked.update(model.key for model in result.models)
            diagnostics.update(result.diagnostics)
            models = _merge_health(models, result)
            plans = recommend_plans(profile, models)
        models = _only_verified(models)
        plans = recommend_plans(profile, models)
    return profile, models, plans, diagnostics


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
) -> dict[str, Any]:
    return {
        "task_profile": asdict(profile),
        "available_models": [model_to_dict(model) for model in models if model.healthy is not False],
        "plans": [plan_to_dict(plan) for plan in plans],
        "health_diagnostics": diagnostics,
    }


def _print_recommendation(
    profile: TaskProfile,
    models: list[ModelSpec],
    plans: list[Plan],
    diagnostics: dict[str, str],
    as_json: bool,
) -> None:
    if as_json:
        print(
            json.dumps(
                _recommendation_payload(profile, models, plans, diagnostics),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        f"Task: type={profile.kind}, complexity={profile.complexity}/5, "
        f"risk={profile.risk}/5, tools={'yes' if profile.needs_tools else 'no'}"
    )
    if diagnostics:
        print("\nHealth probes:")
        for key, status in diagnostics.items():
            print(f"  - {key}: {status}")
    for plan in plans:
        print(f"\n[{plan.id}] {plan.label} — mode={plan.mode}, calls≈{plan.estimated_calls}")
        for participant in plan.participants:
            print(
                f"  - {participant.role}: {participant.model.key} "
                f"[reasoning={participant.reasoning_effort}]"
            )
        print(f"  strengths: {', '.join(plan.strengths)}")
        print(f"  risks: {', '.join(plan.risks)}")


def _result_payload(result: CouncilResult) -> dict[str, Any]:
    return {
        "plan": result.plan_id,
        "final": result.final,
        "anonymous_answers": [
            {"label": label, "text": text} for label, text in result.anonymous_answers
        ],
        "reviews": list(result.reviews),
        "failures": list(result.failures),
        "call_count": result.call_count,
    }


def _backup_hermes_config(executable: str = "hermes") -> Path:
    result = subprocess.run(
        [executable, "config", "path"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        shell=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Could not resolve Hermes config path for backup")
    source = Path(result.stdout.strip().splitlines()[-1])
    if not source.is_file():
        raise RuntimeError(f"Hermes config file does not exist: {source}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = source.with_name(f"{source.name}.model-council-backup-{stamp}")
    shutil.copy2(source, backup)
    return backup


def _install_presets(plans: list[Plan]) -> tuple[Path, dict[str, Any]]:
    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.moa_config import normalize_moa_config, validate_moa_payload
    except ImportError as exc:
        raise RuntimeError("Hermes Python modules are unavailable") from exc
    config = load_config()
    raw_moa = config.get("moa") if isinstance(config, dict) else {}
    new_moa = build_native_moa_config(plans, raw_moa)
    problems = validate_moa_payload(new_moa)
    if problems:
        raise RuntimeError("Invalid generated MoA config: " + "; ".join(problems))
    backup = _backup_hermes_config()
    config["moa"] = normalize_moa_config(new_moa)
    save_config(config)
    return backup, config["moa"]


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
    recommend.add_argument("--timeout", type=int, default=180)
    recommend.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="Execute one recommended plan")
    run.add_argument("task")
    run.add_argument("--plan", choices=("fast", "balanced", "quality"), required=True)
    run.add_argument("--no-probe", action="store_true")
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--yes", action="store_true", help="Confirm the displayed model-call budget")
    run.add_argument("--json", action="store_true")

    install = sub.add_parser("install-presets", help="Install native Hermes MoA presets")
    install.add_argument(
        "--task",
        default="复杂、高风险的通用任务，需要独立分析、工具执行和最终审查",
    )
    install.add_argument("--no-probe", action="store_true")
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
        profile, models, plans, diagnostics = prepare_recommendation(
            args.task, live_probe=args.probe, timeout=args.timeout
        )
        _print_recommendation(profile, models, plans, diagnostics, args.json)
        return 0

    if args.command == "run":
        if not args.yes:
            raise SystemExit("Refusing to execute model calls without --yes")
        profile, models, plans, diagnostics = prepare_recommendation(
            args.task, live_probe=not args.no_probe, timeout=args.timeout
        )
        plan = next(plan for plan in plans if plan.id == args.plan)
        runner = CouncilRunner(HermesInvoker(timeout=args.timeout), max_workers=3)
        result = runner.run(args.task, plan)
        if args.json:
            print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2))
        else:
            print(result.final)
            if result.failures:
                print("\nDegraded participants:", file=sys.stderr)
                for failure in result.failures:
                    print(f"- {failure}", file=sys.stderr)
            print(f"\n[model-council calls: {result.call_count}]", file=sys.stderr)
        return 0

    if args.command == "install-presets":
        if not args.yes:
            raise SystemExit("Refusing to modify Hermes config without --yes")
        profile, models, plans, diagnostics = prepare_recommendation(
            args.task, live_probe=not args.no_probe, timeout=args.timeout
        )
        backup, moa = _install_presets(plans)
        payload = {
            "backup": str(backup),
            "default_preset": moa.get("default_preset"),
            "presets": list((moa.get("presets") or {}).keys()),
            "health_diagnostics": diagnostics,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"Installed native Hermes MoA presets. Backup: {backup}")
            print(f"Default preset: {payload['default_preset']}")
            print("Presets: " + ", ".join(payload["presets"]))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
