"""Live health probes for candidate models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import threading
from typing import Callable

from .hermes_invoker import _redact
from .inventory import ModelSpec


_GLOBAL_PROBE_LOCK = threading.Lock()


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
                if "HEALTH_OK" not in output:
                    raise RuntimeError("unexpected health-check response")
                health[index] = True
                diagnostics[model.key] = "ok"
            except Exception as exc:
                health[index] = False
                diagnostics[model.key] = _safe_error(exc)
    updated = tuple(
        replace(model, healthy=health.get(index, False))
        for index, model in enumerate(models)
    )
    return ProbeResult(updated, diagnostics)
