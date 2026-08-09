"""Regression tests for independent HMC review findings."""
import unittest

from model_council.analysis import TaskProfile
from model_council.cli import plan_to_dict
from model_council.inventory import ModelSpec
from model_council.recommender import (
    Participant,
    Plan,
    canonical_model_identity,
    recommend_plans,
)
from model_council.runner import CouncilRunner, _failure_code


class ReviewRegressionTests(unittest.TestCase):
    @staticmethod
    def _plan(mode, participants, estimated_calls, max_calls=9):
        return Plan(
            mode, mode.title(), mode, tuple(participants),
            estimated_calls, max_calls, (), (),
        )

    def test_task_truncation_alone_degrades_every_success_path(self):
        models = [ModelSpec(f"p{i}", f"model-{i}", f"f{i}") for i in range(4)]
        plans = [
            self._plan("single", [Participant("actor", models[0], "low")], 1, 1),
            self._plan("moa", [
                Participant("advisor-1", models[0], "medium"),
                Participant("aggregator", models[1], "high"),
            ], 2, 3),
            self._plan("council", [
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("chairman", models[3], "high"),
            ], 5, 5),
        ]
        for plan in plans:
            with self.subTest(plan=plan.mode):
                result = CouncilRunner(lambda *_: "ok", max_workers=1).run(
                    "T" * 25000, plan, seed=1
                )
                self.assertTrue(result.task_truncated)
                self.assertTrue(result.degraded)
                self.assertEqual(result.degradation_reason, "task_truncated")

    def test_moa_fallback_source_is_first_successful_advisor(self):
        models = [ModelSpec(f"p{i}", f"model-{i}", f"f{i}") for i in range(3)]
        plan = self._plan("moa", [
            Participant("advisor-1", models[0], "medium"),
            Participant("advisor-2", models[1], "medium"),
            Participant("aggregator", models[2], "high"),
        ], 3, 4)

        def invoke(_model, _prompt, role, _effort):
            if role in {"advisor-1", "aggregator"}:
                raise RuntimeError("failed")
            return "SECOND"

        result = CouncilRunner(invoke, max_workers=1).run("task", plan)
        self.assertEqual(result.final, "SECOND")
        self.assertEqual(result.fallback_source, "advisor-2")
        self.assertEqual(result.degradation_reason, "aggregator_failed")

    def test_chairman_fallback_reason_wins_over_participant_failure(self):
        models = [ModelSpec(f"p{i}", f"model-{i}", f"f{i}") for i in range(4)]
        plan = self._plan("council", [
            Participant("advisor-1", models[0], "high"),
            Participant("advisor-2", models[1], "high"),
            Participant("advisor-3", models[2], "high"),
            Participant("chairman", models[3], "high"),
        ], 6, 9)

        def invoke(_model, _prompt, role, _effort):
            if role in {"advisor-1", "chairman"}:
                raise RuntimeError("failed")
            return "ok"

        result = CouncilRunner(invoke, max_workers=1).run("task", plan, seed=1)
        self.assertIsNotNone(result.fallback_source)
        self.assertEqual(result.degradation_reason, "chairman_failed")

    def test_failure_code_uses_fixed_allowlist_fallback(self):
        OpenAIError = type("OpenAIError", (Exception,), {})
        AnthropicAPIError = type("AnthropicAPIError", (Exception,), {})
        self.assertEqual(_failure_code(OpenAIError("backend exploded")), "invocation_error")
        self.assertEqual(
            _failure_code(AnthropicAPIError("backend exploded")), "invocation_error"
        )

    def test_plan_json_includes_structured_degradation(self):
        model = ModelSpec("provider", "only-model", "family", healthy=True)
        plan = recommend_plans(
            TaskProfile("code", 4, 5, True, False, True), [model]
        )[2]
        payload = plan_to_dict(plan)
        self.assertTrue(payload["degraded"])
        self.assertEqual(payload["degradation_reason"], "single_model_inventory")

    def test_partial_reviewer_failure_records_coverage(self):
        models = [ModelSpec(f"p{i}", f"model-{i}", f"f{i}") for i in range(3)]
        plan = self._plan("council", [
            Participant("advisor-1", models[0], "high"),
            Participant("advisor-2", models[1], "high"),
            Participant("chairman", models[2], "high"),
        ], 5, 5)

        def invoke(_model, _prompt, role, _effort):
            if role == "reviewer-1":
                raise TimeoutError("timeout")
            return "ok"

        result = CouncilRunner(invoke, max_workers=1).run("task", plan, seed=1)
        self.assertEqual(result.review_coverage, 0.5)
        self.assertEqual(result.degradation_reason, "peer_review_incomplete")

    def test_degraded_single_plan_preserves_structured_metadata_and_truncation(self):
        model = ModelSpec("provider", "only-model", "family", healthy=True)
        quality = recommend_plans(
            TaskProfile("code", 4, 5, True, False, True), [model]
        )[2]

        result = CouncilRunner(lambda *_: "final").run("T" * 24000, quality)

        self.assertTrue(result.degraded)
        self.assertEqual(result.degradation_reason, "single_model_inventory")
        self.assertEqual(result.candidate_count, 1)
        self.assertTrue(result.task_truncated)

    def test_partial_advisor_failure_is_structurally_degraded(self):
        models = [
            ModelSpec("p1", "model-1", "f1"),
            ModelSpec("p2", "model-2", "f2"),
            ModelSpec("p3", "model-3", "f3"),
            ModelSpec("pc", "chair", "fc"),
        ]
        plan = Plan(
            "quality",
            "Quality",
            "council",
            tuple(
                [Participant(f"advisor-{i+1}", model, "high") for i, model in enumerate(models[:3])]
                + [Participant("chairman", models[3], "high")]
            ),
            6,
            9,
            (),
            (),
        )

        def invoke(model, _prompt, role, _effort):
            if role == "advisor-1":
                raise RuntimeError("temporary")
            if role.startswith("advisor"):
                return "candidate"
            if role.startswith("reviewer"):
                return "review"
            return "final"

        result = CouncilRunner(invoke, max_workers=1).run("task", plan)

        self.assertTrue(result.degraded)
        self.assertEqual(result.degradation_reason, "participant_failure")
        self.assertEqual(result.candidate_count, 2)

    def test_moa_fallback_and_truncation_are_structured(self):
        advisor = ModelSpec("pa", "advisor-model", "fa")
        aggregator = ModelSpec("pg", "aggregator-model", "fg")
        plan = Plan(
            "balanced",
            "Balanced",
            "moa",
            (
                Participant("advisor", advisor, "medium"),
                Participant("aggregator", aggregator, "high"),
            ),
            2,
            3,
            (),
            (),
        )

        def invoke(_model, _prompt, role, _effort):
            if role == "aggregator":
                raise TimeoutError("timeout")
            return "advisor answer"

        result = CouncilRunner(invoke).run("T" * 7000, plan)

        self.assertTrue(result.degraded)
        self.assertEqual(result.degradation_reason, "aggregator_failed")
        self.assertEqual(result.fallback_source, "advisor-1")
        self.assertEqual(result.candidate_count, 1)
        self.assertTrue(result.task_truncated)

    def test_scrubbing_includes_chairman_and_failed_participants(self):
        failed = ModelSpec("failed-provider", "failed-model", "failed-family")
        advisor = ModelSpec("advisor-provider", "advisor-model", "advisor-family")
        chairman = ModelSpec("secret-chair-provider", "secret-chair-model", "chair-family")
        plan = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-1", failed, "high"),
                Participant("advisor-2", advisor, "high"),
                Participant("chairman", chairman, "high"),
            ),
            3,
            5,
            (),
            (),
        )
        reviewer_prompts = []

        def invoke(model, prompt, role, _effort):
            if role == "advisor-1":
                raise RuntimeError("temporary")
            if role == "advisor-2":
                return (
                    "Compare SECRET CHAIR MODEL from SECRET CHAIR PROVIDER "
                    "with FAILED MODEL from FAILED PROVIDER"
                )
            if role.startswith("reviewer"):
                reviewer_prompts.append(prompt)
                return "review"
            return "I am SECRET CHAIR MODEL from SECRET CHAIR PROVIDER"

        result = CouncilRunner(invoke, max_workers=1).run("task", plan)
        exposed = " ".join(text for _, text in result.anonymous_answers).lower()
        exposed += " " + " ".join(reviewer_prompts).lower()
        exposed += " " + result.final.lower()
        for identity in (
            "secret chair model",
            "secret chair provider",
            "failed model",
            "failed provider",
        ):
            self.assertNotIn(identity, exposed)

    def test_cosmetic_aliases_share_conservative_identity(self):
        alias_a = ModelSpec("provider-a", "shared-model", "family-a", healthy=True)
        alias_b = ModelSpec("provider-b", "SHARED_MODEL", "family-b", healthy=True)
        self.assertEqual(
            canonical_model_identity(alias_a), canonical_model_identity(alias_b)
        )

        models = [
            alias_a,
            alias_b,
            ModelSpec("provider-c", "distinct-model", "family-c", healthy=True),
            ModelSpec("provider-d", "chair-model", "family-d", healthy=True),
        ]
        quality = recommend_plans(
            TaskProfile("decision", 5, 5, False, False, True), models
        )[2]
        identities = [
            canonical_model_identity(participant.model)
            for participant in quality.participants
            if participant.role.startswith("advisor")
        ]
        self.assertEqual(len(identities), len(set(identities)))

    def test_budget_preflight_uses_same_alias_topology_as_runtime(self):
        alias_a = ModelSpec("provider-a", "shared-model", "family-a")
        alias_b = ModelSpec("provider-b", "SHARED_MODEL", "family-b")
        chairman = ModelSpec("provider-c", "chair-model", "family-c")
        plan = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-1", alias_a, "high"),
                Participant("advisor-2", alias_b, "high"),
                Participant("chairman", chairman, "high"),
            ),
            5,
            3,
            (),
            (),
        )
        roles = []

        def invoke(_model, _prompt, role, _effort):
            roles.append(role)
            return "answer"

        result = CouncilRunner(invoke, max_workers=1).run("task", plan)
        self.assertEqual(result.call_count, 3)
        self.assertFalse(any(role.startswith("reviewer") for role in roles))


if __name__ == "__main__":
    unittest.main()
