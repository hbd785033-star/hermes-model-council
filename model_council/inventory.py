"""Read the model inventory that Hermes already exposes to its pickers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    family: str
    is_current: bool = False
    reasoning: bool = False
    fast: bool = False
    healthy: bool | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


def _family(provider: str, model: str) -> str:
    text = f"{provider}/{model}".lower()
    if "anthropic" in text or "claude" in text:
        return "anthropic"
    if "openai" in text or "gpt" in text or "codex" in text:
        return "openai"
    if "gemini" in text or "google" in text:
        return "google"
    if "deepseek" in text:
        return "deepseek"
    if "grok" in text or provider.lower() in {"xai", "xai-oauth"}:
        return "xai"
    return provider.lower()


def _hermes_payload() -> dict[str, Any]:
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context
    except ImportError as exc:
        raise RuntimeError(
            "Hermes Python modules are unavailable. Run this script from a Hermes terminal."
        ) from exc
    return build_models_payload(
        load_picker_context(),
        explicit_only=True,
        include_unconfigured=False,
        picker_hints=True,
        capabilities=True,
        max_models=200,
        probe_custom_providers=False,
        for_picker=True,
    )


def discover_models(*, payload: dict[str, Any] | None = None) -> list[ModelSpec]:
    """Return configured, selectable non-MoA models without exposing credentials."""
    data = payload if payload is not None else _hermes_payload()
    current_provider = str(data.get("provider") or "")
    current_model = str(data.get("model") or "")
    found: list[ModelSpec] = []
    seen: set[tuple[str, str]] = set()
    for row in data.get("providers") or []:
        provider = str(row.get("slug") or "").strip()
        if not provider or provider.lower() == "moa" or row.get("authenticated") is False:
            continue
        capabilities = row.get("capabilities") or {}
        for raw_model in row.get("models") or []:
            model = str(raw_model or "").strip()
            identity = (provider, model)
            if not model or identity in seen:
                continue
            seen.add(identity)
            caps = capabilities.get(model) or {}
            found.append(
                ModelSpec(
                    provider=provider,
                    model=model,
                    family=_family(provider, model),
                    is_current=provider == current_provider and model == current_model,
                    reasoning=bool(caps.get("reasoning")),
                    fast=bool(caps.get("fast")),
                )
            )
    return found
