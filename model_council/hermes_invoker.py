"""Safe subprocess adapter for isolated Hermes model calls."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable

from .inventory import ModelSpec

_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]+"),
        "Bearer [REDACTED]",
    ),
    (
        re.compile(r"(?i)(https?://[^:/\s]+:)[^@\s/]+(@)"),
        r"\1[REDACTED]\2",
    ),
    (
        re.compile(
            r"(?i)((?:[a-z0-9_]*(?:api[_-]?key|token|password|passwd|client[_-]?secret|secret)[a-z0-9_]*)\s*[=:]\s*)[^\s,;]+"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED]"),
)


def _redact(text: str) -> str:
    value = str(text or "")
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


class HermesInvoker:
    """Invoke one model through Hermes without shell interpolation or tool access."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        timeout: int = 240,
        max_prompt_chars: int = 24000,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        resolved = executable or shutil.which("hermes")
        if not resolved:
            raise RuntimeError("The `hermes` executable was not found on PATH")
        self.executable = resolved
        self.timeout = max(1, int(timeout))
        self.max_prompt_chars = max(1000, int(max_prompt_chars))
        self.run_command = run_command

    def __call__(
        self, model: ModelSpec, prompt: str, role: str, reasoning_effort: str
    ) -> str:
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("prompt must not be empty")
        if len(text) > self.max_prompt_chars:
            raise ValueError(
                f"prompt exceeds safe command limit ({len(text)} > {self.max_prompt_chars} chars)"
            )
        command = [
            self.executable,
            "chat",
            "-Q",
            "-m",
            model.model,
            "--provider",
            model.provider,
            "--reasoning",
            reasoning_effort or "medium",
            "-t",
            "",
            "--max-turns",
            "1",
            "--ignore-rules",
            "--safe-mode",
            "--source",
            "model-council",
            "-q",
            text,
        ]
        try:
            result = self.run_command(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{role} timed out after {self.timeout}s using {model.key}"
            ) from exc
        if result.returncode != 0:
            diagnostic = _redact((result.stderr or result.stdout or "unknown error").strip())
            raise RuntimeError(
                f"{role} failed using {model.key} (exit {result.returncode}): {diagnostic[:1000]}"
            )
        output = str(result.stdout or "").strip()
        if not output:
            raise RuntimeError(f"{role} returned an empty response using {model.key}")
        return output
