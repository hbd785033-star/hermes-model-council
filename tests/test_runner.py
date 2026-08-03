import unittest

from model_council.inventory import ModelSpec
from model_council.recommender import Participant, Plan
from model_council.runner import CouncilRunner


class CouncilRunnerTests(unittest.TestCase):
    def test_runs_anonymous_peer_review_then_chairman(self):
        openai = ModelSpec("openai-codex", "gpt-5.6-sol", "openai", healthy=True)
        claude = ModelSpec("anthropic", "claude-opus-4-6", "anthropic", healthy=True)
        plan = Plan(
            id="quality",
            label="Quality",
            mode="council",
            participants=(
                Participant("advisor-1", openai, "high"),
                Participant("advisor-2", claude, "high"),
                Participant("chairman", claude, "high"),
            ),
            estimated_calls=5,
            max_calls=6,
            strengths=(),
            risks=(),
        )
        calls = []

        def invoke(model, prompt, role, effort):
            calls.append((model, prompt, role, effort))
            if role.startswith("advisor"):
                return f"independent answer {role[-1]}"
            if role.startswith("reviewer"):
                return "anonymous review"
            return "FINAL VERDICT"

        result = CouncilRunner(invoke=invoke, max_workers=4).run(
            "Choose a database architecture", plan, seed=7
        )

        self.assertEqual(result.final, "FINAL VERDICT")
        self.assertEqual(len(result.anonymous_answers), 2)
        self.assertEqual(len(result.reviews), 2)
        review_prompts = [prompt for _, prompt, role, _ in calls if role.startswith("reviewer")]
        self.assertTrue(all("Response A" in prompt and "Response B" in prompt for prompt in review_prompts))
        self.assertTrue(all("gpt-5.6-sol" not in prompt for prompt in review_prompts))
        self.assertTrue(all("claude-opus-4-6" not in prompt for prompt in review_prompts))
        self.assertEqual(result.call_count, 5)
        self.assertEqual(result.failures, ())

    def test_identity_scrubbing_is_case_insensitive(self):
        models = [
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic"),
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
        ]
        text = "I AM Claude-Opus-4-6 by ANTHROPIC; compare with GpT-5.6-SoL from OpenAI-Codex."

        cleaned = CouncilRunner._scrub_identities(text, models)

        lowered = cleaned.lower()
        for identity in (
            "claude-opus-4-6",
            "anthropic",
            "gpt-5.6-sol",
            "openai-codex",
        ):
            self.assertNotIn(identity, lowered)
        self.assertIn("[model identity hidden]", cleaned)

    def test_moa_falls_back_to_advisor_when_aggregator_fails(self):
        advisor = ModelSpec("deepseek", "deepseek-v4-pro", "deepseek", healthy=True)
        aggregator = ModelSpec(
            "openai-codex", "gpt-5.6-sol", "openai", healthy=True
        )
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

        def invoke(model, prompt, role, effort):
            if role == "aggregator":
                raise RuntimeError("timed out")
            return "ADVISOR FALLBACK"

        result = CouncilRunner(invoke=invoke).run("task", plan)

        self.assertEqual(result.final, "ADVISOR FALLBACK")
        self.assertEqual(result.call_count, 2)
        self.assertTrue(any("aggregator" in failure for failure in result.failures))

    def test_council_falls_back_to_anonymous_answer_when_chairman_fails(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek"),
        ]
        plan = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("chairman", models[0], "high"),
            ),
            5,
            9,
            (),
            (),
        )

        def invoke(model, prompt, role, effort):
            if role == "chairman":
                raise RuntimeError("chairman timed out")
            if role == "peer-reviewer":
                return "anonymous review"
            return f"answer from {model.model}"

        result = CouncilRunner(invoke=invoke, max_workers=2).run("task", plan)

        self.assertTrue(result.final.startswith("Candidate "))
        self.assertEqual(result.call_count, 5)
        self.assertTrue(any("chairman" in failure for failure in result.failures))
        self.assertNotIn("gpt-5.6-sol", result.final.lower())
        self.assertNotIn("deepseek-v4-pro", result.final.lower())

    def test_refuses_underreported_topology_call_budget(self):
        models = [
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
            ModelSpec("deepseek", "deepseek-v4-pro", "deepseek"),
        ]
        plan = Plan(
            id="quality",
            label="Underreported",
            mode="council",
            participants=(
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("chairman", models[0], "high"),
            ),
            estimated_calls=1,
            max_calls=1,
            strengths=(),
            risks=(),
        )
        calls = []

        with self.assertRaisesRegex(ValueError, "call budget"):
            CouncilRunner(invoke=lambda *args: calls.append(args) or "unused").run(
                "task", plan
            )

        self.assertEqual(calls, [])

    def test_refuses_plan_over_call_budget(self):
        model = ModelSpec("openai-codex", "gpt-5.6-sol", "openai")
        plan = Plan(
            id="quality",
            label="Quality",
            mode="council",
            participants=(Participant("chairman", model, "high"),),
            estimated_calls=10,
            max_calls=3,
            strengths=(),
            risks=(),
        )

        with self.assertRaisesRegex(ValueError, "call budget"):
            CouncilRunner(invoke=lambda *_: "unused").run("task", plan)


if __name__ == "__main__":
    unittest.main()
