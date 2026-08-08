"""Structured evidence contracts and deterministic evidence gating.

This module deliberately does not fetch sources, execute commands, or ask an
LLM to verify facts. Adapters such as a test runner or citation checker should
produce ``EvidenceArtifact`` values, and ``EvidenceGate`` makes the resulting
hard-gate decision deterministically.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .hermes_invoker import _redact


class ClaimImportance(str, Enum):
    """Whether an unverified claim blocks the evidence gate."""

    REQUIRED = "required"
    SUPPORTING = "supporting"


class EvidenceStatus(str, Enum):
    """Status emitted by an external verifier."""

    VERIFIED = "verified"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Claim:
    """A materially checkable statement made by a candidate."""

    id: str
    text: str
    importance: ClaimImportance = ClaimImportance.SUPPORTING


@dataclass(frozen=True)
class EvidenceArtifact:
    """A provenance-bearing result from a deterministic or external verifier."""

    id: str
    claim_id: str
    kind: str
    source: str
    excerpt: str
    status: EvidenceStatus
    verifier: str


@dataclass(frozen=True)
class EvidenceBundle:
    """Claims and verifier outputs for one decision run."""

    claims: tuple[Claim, ...] = ()
    artifacts: tuple[EvidenceArtifact, ...] = ()

    def __post_init__(self) -> None:
        claim_ids = [claim.id for claim in self.claims]
        if any(not claim_id.strip() for claim_id in claim_ids):
            raise ValueError("claim id must not be empty")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("duplicate claim id")
        if any(not claim.text.strip() for claim in self.claims):
            raise ValueError("claim text must not be empty")

        artifact_ids = [artifact.id for artifact in self.artifacts]
        if any(not artifact_id.strip() for artifact_id in artifact_ids):
            raise ValueError("evidence id must not be empty")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("duplicate evidence id")
        known_claims = set(claim_ids)
        for artifact in self.artifacts:
            if artifact.claim_id not in known_claims:
                raise ValueError(f"evidence references unknown claim: {artifact.claim_id}")
            if not artifact.kind.strip():
                raise ValueError("evidence kind must not be empty")
            if not artifact.source.strip():
                raise ValueError("evidence source must not be empty")
            if not artifact.verifier.strip():
                raise ValueError("evidence verifier must not be empty")
            if not isinstance(artifact.status, EvidenceStatus):
                raise ValueError("evidence status must be an EvidenceStatus")


@dataclass(frozen=True)
class EvidenceGateResult:
    """Deterministic verdict over an evidence bundle."""

    passed: bool
    applicable: bool
    coverage: float
    missing_required_claims: tuple[str, ...] = ()
    failed_required_claims: tuple[str, ...] = ()
    contradictory_claims: tuple[str, ...] = ()
    unresolved_claims: tuple[str, ...] = ()
    untrusted_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verdict without copying raw claims or evidence excerpts."""
        return {
            "passed": self.passed,
            "applicable": self.applicable,
            "coverage": self.coverage,
            "missing_required_claims": list(self.missing_required_claims),
            "failed_required_claims": list(self.failed_required_claims),
            "contradictory_claims": list(self.contradictory_claims),
            "unresolved_claims": list(self.unresolved_claims),
            "untrusted_evidence_ids": list(self.untrusted_evidence_ids),
        }


class EvidenceGate:
    """Apply hard evidence requirements without subjective model judgment."""

    def __init__(self, *, trusted_verifiers: tuple[str, ...] = ()) -> None:
        self.trusted_verifiers = {
            str(verifier).strip()
            for verifier in trusted_verifiers
            if str(verifier).strip()
        }

    def evaluate(self, bundle: EvidenceBundle) -> EvidenceGateResult:
        if not bundle.claims:
            return EvidenceGateResult(passed=True, applicable=False, coverage=0.0)

        artifacts_by_claim: dict[str, list[EvidenceArtifact]] = {
            claim.id: [] for claim in bundle.claims
        }
        untrusted_evidence_ids: list[str] = []
        for artifact in bundle.artifacts:
            if artifact.verifier not in self.trusted_verifiers:
                untrusted_evidence_ids.append(artifact.id)
                continue
            artifacts_by_claim[artifact.claim_id].append(artifact)

        verified_ids = {
            claim_id
            for claim_id, artifacts in artifacts_by_claim.items()
            if any(artifact.status is EvidenceStatus.VERIFIED for artifact in artifacts)
        }
        coverage = len(verified_ids) / len(bundle.claims)
        missing_required: list[str] = []
        failed_required: list[str] = []
        contradictory: list[str] = []
        unresolved: list[str] = []

        for claim in bundle.claims:
            artifacts = artifacts_by_claim[claim.id]
            statuses = {artifact.status for artifact in artifacts}
            has_verified = claim.id in verified_ids
            has_failed = EvidenceStatus.FAILED in statuses
            if not has_verified:
                unresolved.append(claim.id)
            if claim.importance is ClaimImportance.REQUIRED and not has_verified:
                missing_required.append(claim.id)
            if claim.importance is ClaimImportance.REQUIRED and has_failed:
                failed_required.append(claim.id)
            if has_verified and has_failed:
                contradictory.append(claim.id)

        passed = not missing_required and not failed_required and not contradictory
        return EvidenceGateResult(
            passed=passed,
            applicable=True,
            coverage=coverage,
            missing_required_claims=tuple(missing_required),
            failed_required_claims=tuple(failed_required),
            contradictory_claims=tuple(contradictory),
            unresolved_claims=tuple(unresolved),
            untrusted_evidence_ids=tuple(untrusted_evidence_ids),
        )


class CommandVerifier:
    """Run a caller-approved command as evidence, without shell interpolation.

    This is an execution boundary, not a sandbox. ``argv`` must come from trusted
    application configuration, never from a model response or untrusted task
    text. The executable allowlist and root guard prevent accidental expansion
    beyond that preconfigured scope.
    """

    def __init__(
        self,
        *,
        root: Path,
        allowed_executables: tuple[str, ...],
        timeout: int = 120,
        max_excerpt_chars: int = 2000,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"verifier root is not a directory: {self.root}")
        self.allowed_executables: dict[str, Path] = {}
        self.allowed_paths: set[str] = set()
        for executable in allowed_executables:
            raw = str(executable).strip()
            if not raw:
                continue
            candidate = Path(raw)
            if candidate.is_absolute():
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError as exc:
                    raise ValueError(f"allowed executable does not exist: {raw}") from exc
            else:
                if candidate.parent != Path("."):
                    raise ValueError("allowed executable must be a name or absolute path")
                located = shutil.which(raw)
                if not located:
                    raise ValueError(f"allowed executable was not found on PATH: {raw}")
                resolved = Path(located).resolve()
            if not resolved.is_file():
                raise ValueError(f"allowed executable is not a file: {resolved}")
            self.allowed_executables[candidate.name.casefold()] = resolved
            self.allowed_paths.add(self._path_key(resolved))
        if not self.allowed_executables:
            raise ValueError("at least one allowed executable is required")
        self.timeout = max(1, int(timeout))
        self.max_excerpt_chars = max(200, int(max_excerpt_chars))

    @staticmethod
    def _path_key(path: Path) -> str:
        value = str(path.resolve())
        return value.casefold() if os.name == "nt" else value

    def _resolve_cwd(self, cwd: str | Path | None) -> Path:
        candidate = self.root if cwd is None else Path(cwd)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("command cwd is outside verifier root")
        if not resolved.is_dir():
            raise ValueError(f"command cwd is not a directory: {resolved}")
        return resolved

    def verify(
        self,
        evidence_id: str,
        claim_id: str,
        argv: tuple[str, ...],
        *,
        cwd: str | Path | None = None,
    ) -> EvidenceArtifact:
        if not argv or not str(argv[0]).strip():
            raise ValueError("command argv must not be empty")
        requested = Path(argv[0])
        executable = requested.name.casefold()
        if requested.is_absolute():
            resolved_executable = requested.resolve()
            if self._path_key(resolved_executable) not in self.allowed_paths:
                raise ValueError(f"command executable is not allowed: {executable}")
        elif requested.parent != Path("."):
            raise ValueError(f"command executable is not allowed: {executable}")
        else:
            configured_executable = self.allowed_executables.get(executable)
            if configured_executable is None:
                raise ValueError(f"command executable is not allowed: {executable}")
            resolved_executable = configured_executable
        workdir = self._resolve_cwd(cwd)
        command = [str(resolved_executable), *argv[1:]]
        try:
            result = subprocess.run(
                command,
                cwd=workdir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            status = EvidenceStatus.UNAVAILABLE
            excerpt = f"timeout={self.timeout}s"
        else:
            status = (
                EvidenceStatus.VERIFIED
                if result.returncode == 0
                else EvidenceStatus.FAILED
            )
            stdout = _redact(str(result.stdout or "").strip())
            stderr = _redact(str(result.stderr or "").strip())
            excerpt = f"exit={result.returncode}"
            if stdout:
                excerpt += f"\nstdout:\n{stdout}"
            if stderr:
                excerpt += f"\nstderr:\n{stderr}"
            excerpt = excerpt[: self.max_excerpt_chars]
        relative_cwd = workdir.relative_to(self.root).as_posix() or "."
        return EvidenceArtifact(
            id=evidence_id,
            claim_id=claim_id,
            kind="command-result",
            source=f"{executable} @ {relative_cwd}",
            excerpt=excerpt,
            status=status,
            verifier=f"command:{executable}",
        )


@dataclass(frozen=True)
class CitationFetchResult:
    status_code: int
    content_type: str
    text: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    addresses = {
        str(row[4][0])
        for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    }
    return tuple(sorted(addresses))


def _fetch_citation(url: str, timeout: int, max_bytes: int) -> CitationFetchResult:
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hermes-model-council/0.1 citation-verifier"},
        method="GET",
    )
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("citation response exceeds byte limit")
        content_type = str(response.headers.get_content_type() or "")
        charset = response.headers.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace")
        return CitationFetchResult(int(response.status), content_type, text)


class CitationVerifier:
    """Verify an exact quote on a caller-approved public HTTPS source."""

    _TEXT_TYPES = {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }

    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...],
        timeout: int = 20,
        max_bytes: int = 262_144,
        fetch: Callable[[str, int, int], CitationFetchResult] = _fetch_citation,
        resolver: Callable[[str, int], tuple[str, ...]] = _resolve_public_addresses,
    ) -> None:
        self.allowed_hosts = {
            str(host).strip().casefold()
            for host in allowed_hosts
            if str(host).strip()
        }
        if not self.allowed_hosts:
            raise ValueError("at least one allowed citation host is required")
        self.timeout = max(1, int(timeout))
        self.max_bytes = max(1024, int(max_bytes))
        self.fetch = fetch
        self.resolver = resolver

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(str(text or "").casefold().split())

    def _validate_url(self, url: str) -> tuple[str, str]:
        parsed = urlsplit(str(url or "").strip())
        if parsed.scheme.casefold() != "https":
            raise ValueError("citation URL must use HTTPS")
        if parsed.username or parsed.password:
            raise ValueError("citation URL userinfo is not allowed")
        if parsed.query or parsed.fragment:
            raise ValueError("citation URL query and fragment are not allowed")
        host = str(parsed.hostname or "").casefold()
        if not host or host not in self.allowed_hosts:
            raise ValueError(f"citation host is not allowed: {host or '[missing]'}")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError("citation host must not be an IP literal")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("citation URL has an invalid port") from exc
        if port not in (None, 443):
            raise ValueError("citation URL must use the default HTTPS port")
        addresses = self.resolver(host, 443)
        if not addresses:
            raise ValueError("citation host did not resolve")
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as exc:
                raise ValueError("citation resolver returned an invalid address") from exc
            if not parsed_address.is_global:
                raise ValueError("citation host must resolve only to public addresses")
        canonical = parsed.geturl()
        return canonical, host

    def verify(
        self,
        evidence_id: str,
        claim_id: str,
        url: str,
        *,
        expected_excerpt: str,
    ) -> EvidenceArtifact:
        expected = self._normalized(expected_excerpt)
        if not expected:
            raise ValueError("expected citation excerpt must not be empty")
        if len(expected) > 2000:
            raise ValueError("expected citation excerpt is too long")
        canonical, host = self._validate_url(url)
        try:
            fetched = self.fetch(canonical, self.timeout, self.max_bytes)
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            status = EvidenceStatus.UNAVAILABLE
            excerpt = f"citation fetch unavailable: {type(exc).__name__}"
        else:
            content_type = fetched.content_type.casefold().split(";", 1)[0].strip()
            is_text = content_type.startswith("text/") or content_type in self._TEXT_TYPES
            if not 200 <= fetched.status_code < 300:
                status = EvidenceStatus.UNAVAILABLE
                excerpt = f"citation HTTP status={fetched.status_code}"
            elif not is_text:
                status = EvidenceStatus.FAILED
                excerpt = "citation content type is not textual"
            elif expected in self._normalized(fetched.text):
                status = EvidenceStatus.VERIFIED
                excerpt = "expected excerpt found in allowed citation"
            else:
                status = EvidenceStatus.FAILED
                excerpt = "expected excerpt not found in allowed citation"
        return EvidenceArtifact(
            id=evidence_id,
            claim_id=claim_id,
            kind="citation",
            source=canonical,
            excerpt=excerpt,
            status=status,
            verifier=f"citation:{host}",
        )
