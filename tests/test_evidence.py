import sys
import tempfile
import unittest
from pathlib import Path

from model_council import EvidenceGate as PublicEvidenceGate
from model_council.evidence import (
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


if __name__ == "__main__":
    unittest.main()
