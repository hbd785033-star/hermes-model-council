import unittest

from model_council.inventory import ModelSpec
from model_council.recommender import Participant, Plan
from model_council.runner import CouncilRunner, _failure_code


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
            calls.append((model.key, prompt, role, effort))
            if role.startswith("advisor"):
                self.assertIn("800 words", prompt)
                return f"draft from {model.model} by {model.provider}"
            if role.startswith("reviewer"):
                self.assertIn("800 words", prompt)
                return "review"
            self.assertIn("1,200 words", prompt)
            return "FINAL VERDICT"

        result = CouncilRunner(invoke=invoke, max_workers=4).run(
            "Choose a database architecture", plan, seed=7
        )

        self.assertEqual(result.final, "FINAL VERDICT")
        self.assertEqual(len(result.anonymous_answers), 2)
        self.assertEqual(len(result.reviews), 2)
        review_prompts = [prompt for _, prompt, role, _ in calls if role.startswith("reviewer")]
        self.assertTrue(all("Response " in prompt for prompt in review_prompts))
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

    def test_failure_diagnostics_do_not_expose_model_identity(self):
        models = [
            ModelSpec("provider-a", "secret-model-a", "family-a"),
            ModelSpec("provider-b", "secret-model-b", "family-b"),
        ]
        plan = Plan(
            "quality", "Quality", "council",
            (
                Participant("advisor-1", models[0], "low"),
                Participant("advisor-2", models[1], "low"),
                Participant("chairman", models[1], "low"),
            ),
            5, 9, (), (),
        )

        def invoke(model, prompt, role, effort):
            if model.key == models[0].key:
                raise RuntimeError("temporary")
            return "answer"

        result = CouncilRunner(
            invoke=invoke, max_workers=1
        ).run("task", plan)

        self.assertTrue(any("advisor-1" in failure for failure in result.failures))
        failures = " ".join(result.failures)
        self.assertNotIn(models[0].key, failures)
        self.assertNotIn(models[1].key, failures)

    def test_failure_diagnostics_use_safe_reason_codes(self):
        self.assertEqual(_failure_code(RuntimeError("timed out after 240s using secret:model")), "timeout")
        self.assertEqual(_failure_code(RuntimeError("provider returned HTTP 429")), "rate_limited")
        self.assertEqual(_failure_code(RuntimeError("authentication failed using secret:model")), "authentication")
        self.assertEqual(_failure_code(RuntimeError("unexpected provider failure")), "runtimeerror")

    def test_single_path_clips_task_before_fixed_prompt_suffix(self):
        model = ModelSpec("provider-a", "model-a", "family-a")
        plan = Plan(
            "fast", "Fast", "single",
            (Participant("actor", model, "low"),),
            1, 1, (), (),
        )
        prompts = []

        def invoke(model, prompt, role, effort):
            prompts.append(prompt)
            self.assertLessEqual(len(prompt), 24000)
            return "FINAL"

        result = CouncilRunner(invoke=invoke).run("T" * 24000, plan)

        self.assertEqual(result.final, "FINAL")
        self.assertEqual(len(prompts), 1)

    def test_peer_review_outputs_are_scrubbed_before_chairman(self):
        models = [
            ModelSpec("anthropic", "claude-opus-4-6", "anthropic"),
            ModelSpec("openai-codex", "gpt-5.6-sol", "openai"),
        ]
        plan = Plan(
            "quality",
            "Quality",
            "council",
            (
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("chairman", models[1], "high"),
            ),
            5,
            9,
            (),
            (),
        )
        chairman_prompts = []

        def invoke(model, prompt, role, effort):
            if role.startswith("advisor"):
                return "anonymous candidate"
            if role.startswith("reviewer"):
                return "I am Claude from ANTHROPIC; GPT-5.6-SOL is weaker."
            chairman_prompts.append(prompt)
            return "final"

        result = CouncilRunner(invoke=invoke, max_workers=1).run("task", plan, seed=1)

        review_text = " ".join(result.reviews).lower()
        chairman_text = " ".join(chairman_prompts).lower()
        for identity in ("claude", "anthropic", "gpt-5.6-sol"):
            self.assertNotIn(identity, review_text)
            self.assertNotIn(identity, chairman_text)

    def test_each_peer_reviewer_does_not_review_its_own_answer(self):
        models = [
            ModelSpec("provider-a", "model-a", "family-a"),
            ModelSpec("provider-b", "model-b", "family-b"),
        ]
        plan = Plan(
            "quality", "Quality", "council",
            (
                Participant("advisor-1", models[0], "high"),
                Participant("advisor-2", models[1], "high"),
                Participant("chairman", models[0], "high"),
            ),
            5, 9, (), (),
        )
        review_prompts = []

        def invoke(model, prompt, role, effort):
            if role.startswith("advisor"):
                return "ANSWER_A" if model.key == models[0].key else "ANSWER_B"
            if role.startswith("reviewer"):
                review_prompts.append((model.key, prompt))
                return "review"
            return "final"

        CouncilRunner(invoke=invoke, max_workers=1).run("task", plan, seed=1)

        self.assertEqual(len(review_prompts), 2)
        for model_key, prompt in review_prompts:
            own_answer = "ANSWER_A" if model_key == models[0].key else "ANSWER_B"
            other_answer = "ANSWER_B" if model_key == models[0].key else "ANSWER_A"
            self.assertNotIn(own_answer, prompt)
            self.assertIn(other_answer, prompt)

    def test_council_bounds_intermediate_outputs_to_prompt_limit(self):
        models = [
            ModelSpec("provider-a", "model-a", "family-a"),
            ModelSpec("provider-b", "model-b", "family-b"),
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
            if len(prompt) > 24000:
                raise ValueError("prompt exceeds safe command limit")
            if role.startswith("advisor"):
                return "A" * 16000
            if role.startswith("reviewer"):
                return "R" * 12000
            return "FINAL"

        result = CouncilRunner(invoke=invoke, max_workers=1).run("task", plan)

        self.assertEqual(result.final, "FINAL")
        self.assertEqual(result.failures, ())
        self.assertEqual(result.call_count, 5)

    def test_council_bounds_long_task_before_advisor_calls(self):
        models = [
            ModelSpec("provider-a", "model-a", "family-a"),
            ModelSpec("provider-b", "model-b", "family-b"),
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
            if len(prompt) > 24000:
                raise ValueError("prompt exceeds safe command limit")
            if role.startswith("advisor"):
                return "candidate"
            if role.startswith("reviewer"):
                return "review"
            return "FINAL"

        result = CouncilRunner(invoke=invoke, max_workers=1).run("T" * 23900, plan)

        self.assertEqual(result.final, "FINAL")
        self.assertEqual(result.failures, ())

    def test_bounded_block_never_exceeds_requested_limit(self):
        block = CouncilRunner._bounded_block(
            [("One", "A" * 100), ("Two", "B" * 100), ("Three", "C" * 100)],
            120,
        )

        self.assertLessEqual(len(block), 120)

    def test_scrubs_short_provider_identity(self):
        model = ModelSpec("xai", "grok-4", "xai")

        cleaned = CouncilRunner._scrub_identities("Built by xAI", [model])

        self.assertNotIn("xai", cleaned.lower())

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
