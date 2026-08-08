import sys
import tempfile
import unittest
from pathlib import Path

from model_council import EvidenceGate as PublicEvidenceGate
from model_council.evidence import (
    CitationFetchResult,
    CitationVerifier,
    Claim,
    ClaimImportance,
    CommandVerifier,
    EvidenceArtifact,
    EvidenceBundle,
    EvidenceGate,
    EvidenceStatus,
)


class EvidenceGateTests(unittest.TestCase):
    def test_evidence_gate_is_exported_from_package_api(self):
        self.assertIs(PublicEvidenceGate, EvidenceGate)

    def test_untrusted_self_report_cannot_satisfy_required_claim(self):
        forged = EvidenceArtifact(
            "forged",
            "security",
            "model-output",
            "candidate-answer",
            "I verified this myself",
            EvidenceStatus.VERIFIED,
            "model-self-report",
        )
        bundle = EvidenceBundle(
            claims=(Claim("security", "The design is secure", ClaimImportance.REQUIRED),),
            artifacts=(forged,),
        )

        result = EvidenceGate(trusted_verifiers=("pytest",)).evaluate(bundle)

        self.assertFalse(result.passed)
        self.assertEqual(result.missing_required_claims, ("security",))
        self.assertEqual(result.untrusted_evidence_ids, ("forged",))

    def test_passes_when_required_claims_have_verified_evidence(self):
        bundle = EvidenceBundle(
            claims=(
                Claim("security", "The token is stored in an HttpOnly cookie", ClaimImportance.REQUIRED),
                Claim("cost", "The deployment uses one database", ClaimImportance.SUPPORTING),
            ),
            artifacts=(
                EvidenceArtifact(
                    "test-1",
                    "security",
                    "test-result",
                    "tests/test_auth.py:42",
                    "test passed",
                    EvidenceStatus.VERIFIED,
                    "pytest",
                ),
            ),
        )

        result = EvidenceGate(trusted_verifiers=("pytest",)).evaluate(bundle)

        self.assertTrue(result.applicable)
        self.assertTrue(result.passed)
        self.assertEqual(result.coverage, 0.5)
        self.assertEqual(result.missing_required_claims, ())
        self.assertEqual(result.unresolved_claims, ("cost",))

    def test_rejects_missing_required_evidence(self):
        bundle = EvidenceBundle(
            claims=(Claim("security", "The design prevents token replay", ClaimImportance.REQUIRED),),
            artifacts=(),
        )

        result = EvidenceGate().evaluate(bundle)

        self.assertTrue(result.applicable)
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_required_claims, ("security",))
        self.assertEqual(result.failed_required_claims, ())

    def test_rejects_failed_required_evidence_and_reports_contradiction(self):
        bundle = EvidenceBundle(
            claims=(
                Claim("security", "The endpoint rejects expired tokens", ClaimImportance.REQUIRED),
                Claim("latency", "The endpoint responds within 100ms", ClaimImportance.SUPPORTING),
            ),
            artifacts=(
                EvidenceArtifact("bad", "security", "test-result", "test_auth.py", "failed", EvidenceStatus.FAILED, "pytest"),
                EvidenceArtifact("ok", "security", "test-result", "test_auth.py", "passed", EvidenceStatus.VERIFIED, "pytest"),
                EvidenceArtifact("latency-unknown", "latency", "benchmark", "bench.log", "not run", EvidenceStatus.UNAVAILABLE, "bench"),
            ),
        )

        result = EvidenceGate(trusted_verifiers=("pytest", "bench")).evaluate(bundle)

        self.assertFalse(result.passed)
        self.assertEqual(result.failed_required_claims, ("security",))
        self.assertEqual(result.contradictory_claims, ("security",))
        self.assertEqual(result.unresolved_claims, ("latency",))

    def test_empty_bundle_is_not_applicable(self):
        result = EvidenceGate().evaluate(EvidenceBundle())

        self.assertFalse(result.applicable)
        self.assertTrue(result.passed)
        self.assertEqual(result.coverage, 0.0)

    def test_rejects_duplicate_and_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate claim id"):
            EvidenceBundle(
                claims=(Claim("c", "one"), Claim("c", "two")),
            )

        with self.assertRaisesRegex(ValueError, "unknown claim"):
            EvidenceBundle(
                claims=(Claim("c", "one"),),
                artifacts=(EvidenceArtifact("e", "missing", "test", "x", "", EvidenceStatus.VERIFIED, "pytest"),),
            )

    def test_serializes_verdict_without_raw_secrets(self):
        bundle = EvidenceBundle(
            claims=(Claim("c", "The check passed", ClaimImportance.REQUIRED),),
            artifacts=(EvidenceArtifact("e", "c", "test", "tests/test_x.py", "passed", EvidenceStatus.VERIFIED, "pytest"),),
        )

        payload = EvidenceGate(trusted_verifiers=("pytest",)).evaluate(bundle).to_dict()

        self.assertEqual(payload["passed"], True)
        self.assertEqual(payload["coverage"], 1.0)
        self.assertEqual(payload["missing_required_claims"], [])
        self.assertNotIn("api_key", str(payload).lower())

    def test_command_verifier_runs_allowed_command_and_gates_result(self):
        with tempfile.TemporaryDirectory() as directory:
            verifier = CommandVerifier(
                root=Path(directory),
                allowed_executables=(Path(sys.executable).name,),
                timeout=5,
            )
            artifact = verifier.verify(
                evidence_id="test-run",
                claim_id="tests-pass",
                argv=(sys.executable, "-c", "print('verification passed')"),
            )
            bundle = EvidenceBundle(
                claims=(Claim("tests-pass", "Tests pass", ClaimImportance.REQUIRED),),
                artifacts=(artifact,),
            )

        self.assertEqual(artifact.status, EvidenceStatus.VERIFIED)
        self.assertIn("verification passed", artifact.excerpt)
        self.assertTrue(
            EvidenceGate(trusted_verifiers=(artifact.verifier,)).evaluate(bundle).passed
        )

    def test_command_verifier_records_nonzero_exit_as_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = CommandVerifier(
                root=Path(directory),
                allowed_executables=(Path(sys.executable).name,),
            ).verify(
                evidence_id="test-run",
                claim_id="tests-pass",
                argv=(sys.executable, "-c", "raise SystemExit(3)"),
            )

        self.assertEqual(artifact.status, EvidenceStatus.FAILED)
        self.assertIn("exit=3", artifact.excerpt)

    def test_command_verifier_rejects_disallowed_executable_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            verifier = CommandVerifier(
                root=Path(directory),
                allowed_executables=(Path(sys.executable).name,),
            )
            with self.assertRaisesRegex(ValueError, "not allowed"):
                verifier.verify("e", "c", ("definitely-not-allowed", "--version"))
            spoofed = Path(directory) / Path(sys.executable).name
            spoofed.write_text("not the trusted executable", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not allowed"):
                verifier.verify("e", "c", (str(spoofed), "--version"))
            with self.assertRaisesRegex(ValueError, "outside verifier root"):
                verifier.verify("e", "c", (sys.executable, "-V"), cwd="..")

    def test_citation_verifier_requires_public_allowed_https_source_and_quote(self):
        fetched_urls = []

        def fetch(url, timeout, max_bytes):
            fetched_urls.append((url, timeout, max_bytes))
            return CitationFetchResult(
                status_code=200,
                content_type="text/html",
                text="The official policy requires signed requests.",
            )

        verifier = CitationVerifier(
            allowed_hosts=("docs.example",),
            fetch=fetch,
            resolver=lambda host, port: ("93.184.216.34",),
        )
        artifact = verifier.verify(
            "policy-source",
            "signed-requests",
            "https://docs.example/security/policy",
            expected_excerpt="official policy requires signed requests",
        )
        bundle = EvidenceBundle(
            claims=(
                Claim(
                    "signed-requests",
                    "Requests are signed",
                    ClaimImportance.REQUIRED,
                ),
            ),
            artifacts=(artifact,),
        )

        self.assertEqual(artifact.status, EvidenceStatus.VERIFIED)
        self.assertEqual(fetched_urls[0][0], "https://docs.example/security/policy")
        self.assertTrue(
            EvidenceGate(trusted_verifiers=(artifact.verifier,)).evaluate(bundle).passed
        )

    def test_citation_verifier_marks_missing_quote_as_failed_evidence(self):
        verifier = CitationVerifier(
            allowed_hosts=("docs.example",),
            fetch=lambda url, timeout, max_bytes: CitationFetchResult(
                status_code=200,
                content_type="text/plain",
                text="This page discusses a different subject.",
            ),
            resolver=lambda host, port: ("93.184.216.34",),
        )

        artifact = verifier.verify(
            "policy-source",
            "signed-requests",
            "https://docs.example/security/policy",
            expected_excerpt="requests are signed",
        )

        self.assertEqual(artifact.status, EvidenceStatus.FAILED)

    def test_citation_verifier_rejects_untrusted_scheme_host_and_private_dns(self):
        verifier = CitationVerifier(
            allowed_hosts=("docs.example",),
            fetch=lambda url, timeout, max_bytes: self.fail("fetch must not run"),
            resolver=lambda host, port: ("93.184.216.34",),
        )
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            verifier.verify("e", "c", "http://docs.example/policy", expected_excerpt="policy")
        with self.assertRaisesRegex(ValueError, "not allowed"):
            verifier.verify("e", "c", "https://evil.example/policy", expected_excerpt="policy")

        private_dns = CitationVerifier(
            allowed_hosts=("docs.example",),
            fetch=lambda url, timeout, max_bytes: self.fail("fetch must not run"),
            resolver=lambda host, port: ("127.0.0.1",),
        )
        with self.assertRaisesRegex(ValueError, "public"):
            private_dns.verify(
                "e", "c", "https://docs.example/policy", expected_excerpt="policy"
            )

    def test_citation_verifier_does_not_accept_redirect_binary_or_oversized_response(self):
        def fetch_by_status(url, timeout, max_bytes):
            if url.endswith("/redirect"):
                return CitationFetchResult(302, "text/html", "redirect")
            if url.endswith("/binary"):
                return CitationFetchResult(200, "application/pdf", "policy")
            raise ValueError("citation response exceeds byte limit")

        verifier = CitationVerifier(
            allowed_hosts=("docs.example",),
            fetch=fetch_by_status,
            resolver=lambda host, port: ("93.184.216.34",),
        )

        redirect = verifier.verify(
            "redirect", "claim", "https://docs.example/redirect", expected_excerpt="redirect"
        )
        binary = verifier.verify(
            "binary", "claim", "https://docs.example/binary", expected_excerpt="policy"
        )
        oversized = verifier.verify(
            "oversized", "claim", "https://docs.example/oversized", expected_excerpt="policy"
        )

        self.assertEqual(redirect.status, EvidenceStatus.UNAVAILABLE)
        self.assertEqual(binary.status, EvidenceStatus.FAILED)
        self.assertEqual(oversized.status, EvidenceStatus.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
