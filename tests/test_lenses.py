import unittest

from model_council.lenses import (
    DECISION_LENSES,
    LENS_POLICY_VERSION,
    resolve_advisor_lens,
    select_decision_lenses,
)


class DecisionLensTests(unittest.TestCase):
    def test_advisor_counts_select_exact_lens_prefixes(self):
        expected = {
            0: (),
            1: ("solution",),
            2: ("solution", "risk"),
            3: ("solution", "risk", "feasibility"),
        }

        for advisor_count, lens_ids in expected.items():
            with self.subTest(advisor_count=advisor_count):
                self.assertEqual(
                    tuple(lens.id for lens in select_decision_lenses(advisor_count)),
                    lens_ids,
                )

    def test_three_advisors_receive_all_v1_lenses_in_order(self):
        self.assertEqual(
            tuple(lens.id for lens in select_decision_lenses(3)),
            ("solution", "risk", "feasibility"),
        )

    def test_invalid_advisor_counts_fail_closed(self):
        for advisor_count in (-1, 4):
            with self.subTest(advisor_count=advisor_count):
                with self.assertRaisesRegex(ValueError, "between 0 and 3"):
                    select_decision_lenses(advisor_count)

    def test_v1_lenses_are_unique_non_empty_and_distinct(self):
        self.assertEqual(LENS_POLICY_VERSION, "hmc-lenses-v1.0")
        self.assertEqual(
            tuple(lens.id for lens in DECISION_LENSES),
            ("solution", "risk", "feasibility"),
        )
        self.assertEqual(len({lens.id for lens in DECISION_LENSES}), 3)
        self.assertTrue(all(lens.instruction.strip() for lens in DECISION_LENSES))
        self.assertEqual(len({lens.instruction for lens in DECISION_LENSES}), 3)

    def test_explicit_advisor_roles_resolve_to_matching_lenses(self):
        expected = {
            "advisor-solution": "solution",
            "advisor-risk": "risk",
            "advisor-feasibility": "feasibility",
        }

        for role, lens_id in expected.items():
            with self.subTest(role=role):
                self.assertEqual(resolve_advisor_lens(role).id, lens_id)

    def test_legacy_advisor_roles_resolve_to_v1_lenses(self):
        expected = {
            "advisor": "solution",
            "advisor-1": "solution",
            "advisor-2": "risk",
            "advisor-3": "feasibility",
        }

        for role, lens_id in expected.items():
            with self.subTest(role=role):
                self.assertEqual(resolve_advisor_lens(role).id, lens_id)

    def test_unknown_advisor_role_falls_back_to_solution(self):
        self.assertEqual(resolve_advisor_lens("advisor-custom").id, "solution")

    def test_non_advisor_role_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not an advisor"):
            resolve_advisor_lens("chairman")


if __name__ == "__main__":
    unittest.main()
