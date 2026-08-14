"""Hermes configuration integration and filesystem transaction helpers."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .presets import build_native_moa_config
from .recommender import Plan


def backup_hermes_config(executable: str = "hermes") -> tuple[Path, Path]:
    result = subprocess.run(
        [executable, "config", "path"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        shell=False,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Could not resolve Hermes config path for backup")
    source = Path(result.stdout.strip().splitlines()[-1])
    if not source.is_file():
        raise RuntimeError(f"Hermes config file does not exist: {source}")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    backup = source.with_name(f"{source.name}.model-council-backup-{stamp}")
    shutil.copy2(source, backup)
    return source, backup


def check_hermes_config(executable: str = "hermes") -> None:
    result = subprocess.run(
        [executable, "config", "check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"Hermes config check failed: {diagnostic[:1000]}")


def save_config_with_rollback(
    config: dict[str, Any],
    *,
    save_config: Callable[[dict[str, Any]], Any],
    source: Path,
    backup: Path,
    check_config: Callable[[], None] = check_hermes_config,
) -> None:
    try:
        save_config(config)
        check_config()
    except Exception as exc:
        shutil.copy2(backup, source)
        try:
            check_config()
        except Exception as restore_exc:  # noqa: BLE001 - validation callback boundary
            raise RuntimeError(
                "Generated Hermes config failed validation; backup restoration also failed "
                f"validation: {restore_exc}"
            ) from exc
        raise RuntimeError(
            f"Hermes config update failed and was restored from backup: {exc}"
        ) from exc


def install_native_presets(plans: list[Plan]) -> tuple[Path, dict[str, Any]]:
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
    source, backup = backup_hermes_config()
    normalized_moa = normalize_moa_config(new_moa)
    config["moa"] = normalized_moa
    save_config_with_rollback(
        config,
        save_config=save_config,
        source=source,
        backup=backup,
    )
    return backup, normalized_moa
