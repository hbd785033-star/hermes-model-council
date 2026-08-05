"""Live health probes for candidate models."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from .hermes_invoker import _redact
from .inventory import ModelSpec

_GLOBAL_PROBE_LOCK = threading.Lock()


class HealthCache:
    """Short-lived model health cache without prompts, outputs, or diagnostics."""

    def __init__(
        self,
        path: Path,
        ttl_seconds: int = 900,
        failure_ttl_seconds: int = 120,
    ):
        self.path = Path(path)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.failure_ttl_seconds = max(0, int(failure_ttl_seconds))

    def load(self, models: list[ModelSpec], *, now: float | None = None) -> dict[str, bool]:
        current = time.time() if now is None else float(now)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        allowed = {model.key for model in models}
        entries = payload.get("models") if isinstance(payload, dict) else {}
        if not isinstance(entries, dict):
            return {}
        found: dict[str, bool] = {}
        for key, entry in entries.items():
            if key not in allowed or not isinstance(entry, dict):
                continue
            healthy = entry.get("healthy")
            checked_at = entry.get("checked_at")
            if not isinstance(healthy, bool) or not isinstance(checked_at, (int, float)):
                continue
            ttl = self.ttl_seconds if healthy else self.failure_ttl_seconds
            if 0 <= current - float(checked_at) <= ttl:
                found[key] = healthy
        return found

    def store(self, health: dict[str, bool], *, now: float | None = None) -> None:
        checked_at = time.time() if now is None else float(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing_payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing_payload = {}
        existing_entries = (
            existing_payload.get("models")
            if isinstance(existing_payload, dict)
            else {}
        )
        entries = dict(existing_entries) if isinstance(existing_entries, dict) else {}
        entries.update(
            {
                key: {"healthy": value, "checked_at": checked_at}
                for key, value in health.items()
                if isinstance(value, bool)
            }
        )
        payload = {
            "version": 1,
            "models": {key: entries[key] for key in sorted(entries)},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


@dataclass(frozen=True)
class ProbeResult:
    models: tuple[ModelSpec, ...]
    diagnostics: dict[str, str]


def _safe_error(exc: Exception) -> str:
    return _redact(str(exc or type(exc).__name__))[:500]


def probe_models(
    models: list[ModelSpec],
    *,
    invoke: Callable[[ModelSpec, str, str, str], str],
    max_workers: int = 3,
) -> ProbeResult:
    """Probe models concurrently with one minimal, tool-free completion each."""
    if not models:
        return ProbeResult((), {})
    health: dict[int, bool] = {}
    diagnostics: dict[str, str] = {}

    def invoke_serialized(model: ModelSpec) -> str:
        with _GLOBAL_PROBE_LOCK:
            return invoke(
                model,
                "Configuration health check. Reply with exactly HEALTH_OK.",
                "health-check",
                "low",
            )

    with ThreadPoolExecutor(max_workers=min(max(1, max_workers), len(models))) as pool:
        futures = {
            pool.submit(invoke_serialized, model): (index, model)
            for index, model in enumerate(models)
        }
        for future in as_completed(futures):
            index, model = futures[future]
            try:
                output = str(future.result() or "").strip()
                if output != "HEALTH_OK":
                    raise RuntimeError("unexpected health-check response")
                health[index] = True
                diagnostics[model.key] = "ok"
            except Exception as exc:  # noqa: BLE001 - provider boundary
                health[index] = False
                diagnostics[model.key] = _safe_error(exc)
    updated = tuple(
        replace(model, healthy=health.get(index, False))
        for index, model in enumerate(models)
    )
    return ProbeResult(updated, diagnostics)
