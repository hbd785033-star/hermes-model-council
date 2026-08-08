import unittest

from model_council.entailment import (
    EntailmentAssessment,
    EntailmentPolicy,
    EntailmentVerdict,
    EvaluatorCalibration,
)
from model_council.evidence import (
    Claim,
    ClaimImportance,
    EvidenceArtifact,
    EvidenceBundle,
    EvidenceStatus,
)


class EntailmentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.bundle = EvidenceBundle(
            claims=(Claim("c1", "Requests must be signed", ClaimImportance.REQUIRED),),
            artifacts=(
                EvidenceArtifact(
                    "e1",
                    "c1",
                    "citation",
                    "https://docs.example/policy",
                    "expected excerpt found",
                    EvidenceStatus.VERIFIED,
                    "citation:docs.example",
                ),
            ),
        )

    def test_supported_assessment_is_advisory_without_calibration(self):
        assessment = EntailmentAssessment(
            "a1", "c1", "e1", EntailmentVerdict.SUPPORTED, "judge-a", "direct support"
        )

        result = EntailmentPolicy(trusted_evaluators=("judge-a",)).evaluate(
            self.bundle, (assessment,)
        )

        self.assertEqual(result.claims[0].verdict, EntailmentVerdict.SUPPORTED)
        self.assertTrue(result.advisory)
        self.assertFalse(result.hard_gate_eligible)
        self.assertFalse(result.should_block)

    def test_conflicting_trusted_assessments_report_disagreement(self):
        assessments = (
            EntailmentAssessment("a1", "c1", "e1", EntailmentVerdict.SUPPORTED, "judge-a", "support"),
            EntailmentAssessment("a2", "c1", "e1", EntailmentVerdict.CONTRADICTED, "judge-b", "conflict"),
        )

        result = EntailmentPolicy(
            trusted_evaluators=("judge-a", "judge-b")
        ).evaluate(self.bundle, assessments)

        claim = result.claims[0]
        self.assertEqual(claim.verdict, EntailmentVerdict.INSUFFICIENT)
        self.assertTrue(claim.disagreement)
        self.assertFalse(result.should_block)

    def test_untrusted_model_self_assessment_is_ignored(self):
        forged = EntailmentAssessment(
            "forged", "c1", "e1", EntailmentVerdict.SUPPORTED, "candidate-model", "self report"
        )

        result = EntailmentPolicy(trusted_evaluators=("judge-a",)).evaluate(
            self.bundle, (forged,)
        )

        self.assertEqual(result.claims[0].verdict, EntailmentVerdict.INSUFFICIENT)
        self.assertEqual(result.untrusted_assessment_ids, ("forged",))

    def test_untrusted_assessment_with_unknown_references_cannot_crash_policy(self):
        forged = EntailmentAssessment(
            "forged", "missing-claim", "missing-evidence",
            EntailmentVerdict.SUPPORTED, "candidate-model", "self report"
        )

        result = EntailmentPolicy(trusted_evaluators=("judge-a",)).evaluate(
            self.bundle, (forged,)
        )

        self.assertEqual(result.untrusted_assessment_ids, ("forged",))
        self.assertEqual(result.claims[0].verdict, EntailmentVerdict.INSUFFICIENT)

    def test_uncalibrated_hard_gate_request_remains_advisory(self):
        assessment = EntailmentAssessment(
            "a1", "c1", "e1", EntailmentVerdict.CONTRADICTED, "judge-a", "not entailed"
        )
        weak = EvaluatorCalibration("judge-a", "gold-v1", 20, 0.9, 0.01)

        result = EntailmentPolicy(
            trusted_evaluators=("judge-a",),
            enable_hard_gate=True,
            calibrations=(weak,),
        ).evaluate(self.bundle, (assessment,))

        self.assertTrue(result.advisory)
        self.assertFalse(result.hard_gate_eligible)
        self.assertFalse(result.should_block)

    def test_calibrated_contradiction_can_block_when_explicitly_enabled(self):
        assessment = EntailmentAssessment(
            "a1", "c1", "e1", EntailmentVerdict.CONTRADICTED, "judge-a", "opposite meaning"
        )
        calibrated = EvaluatorCalibration("judge-a", "gold-v1", 150, 0.82, 0.02)

        result = EntailmentPolicy(
            trusted_evaluators=("judge-a",),
            enable_hard_gate=True,
            calibrations=(calibrated,),
        ).evaluate(self.bundle, (assessment,))

        self.assertFalse(result.advisory)
        self.assertTrue(result.hard_gate_eligible)
        self.assertTrue(result.should_block)

    def test_assessment_references_must_exist(self):
        bad = EntailmentAssessment(
            "a1", "missing", "e1", EntailmentVerdict.SUPPORTED, "judge-a", "bad id"
        )
        with self.assertRaisesRegex(ValueError, "unknown claim"):
            EntailmentPolicy(trusted_evaluators=("judge-a",)).evaluate(
                self.bundle, (bad,)
            )


if __name__ == "__main__":
    unittest.main()
