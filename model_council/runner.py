"""Execution engine for single, MoA-style, and anonymous council plans."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import random
import re
from typing import Callable

from .inventory import ModelSpec
from .recommender import Participant, Plan


Invoker = Callable[[ModelSpec, str, str, str], str]


@dataclass(frozen=True)
class CouncilResult:
    plan_id: str
    final: str
    anonymous_answers: tuple[tuple[str, str], ...]
    reviews: tuple[str, ...]
    failures: tuple[str, ...]
    call_count: int


class CouncilRunner:
    def __init__(self, invoke: Invoker, max_workers: int = 4):
        self.invoke = invoke
        self.max_workers = max(1, int(max_workers))

    @staticmethod
    def _required_calls(plan: Plan) -> int:
        if plan.mode == "single":
            return 1
        if plan.mode == "moa":
            return len(plan.participants)
        if plan.mode == "council":
            advisor_count = sum(
                participant.role.startswith("advisor")
                for participant in plan.participants
            )
            return advisor_count + min(2, advisor_count) + 1
        return 0

    def run(self, task: str, plan: Plan, *, seed: int | None = None) -> CouncilResult:
        if not str(task or "").strip():
            raise ValueError("task must not be empty")
        required_calls = self._required_calls(plan)
        budgeted_calls = max(plan.estimated_calls, required_calls)
        if budgeted_calls > plan.max_calls:
            raise ValueError(
                f"Plan exceeds call budget: {budgeted_calls} > {plan.max_calls}"
            )
        if plan.mode == "single":
            return self._run_single(task, plan)
        if plan.mode == "moa":
            return self._run_moa(task, plan)
        if plan.mode == "council":
            return self._run_council(task, plan, seed=seed)
        raise ValueError(f"Unsupported plan mode: {plan.mode}")

    def _run_single(self, task: str, plan: Plan) -> CouncilResult:
        actor = plan.chairman
        final = self.invoke(actor.model, task, "actor", actor.reasoning_effort)
        return CouncilResult(plan.id, final, (), (), (), 1)

    def _parallel(
        self, participants: list[Participant], prompts: list[str], role_prefix: str
    ) -> tuple[list[tuple[Participant, str]], list[str], int]:
        completed: dict[int, tuple[Participant, str]] = {}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(participants))) as pool:
            futures = {
                pool.submit(
                    self.invoke,
                    participant.model,
                    prompt,
                    f"{role_prefix}-{index + 1}",
                    participant.reasoning_effort,
                ): (index, participant)
                for index, (participant, prompt) in enumerate(zip(participants, prompts))
            }
            for future in as_completed(futures):
                index, participant = futures[future]
                try:
                    text = str(future.result() or "").strip()
                    if not text:
                        raise RuntimeError("empty response")
                    completed[index] = (participant, text)
                except Exception as exc:
                    failures.append(
                        f"{role_prefix}-{index + 1} ({participant.model.key}) failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        return [completed[index] for index in sorted(completed)], failures, len(participants)

    @staticmethod
    def _advisor_prompt(task: str, role: str) -> str:
        return (
            "You are an independent member of a model council. "
            f"Your assigned lens is {role}. Analyze independently and do not imitate consensus.\n\n"
            f"TASK:\n{task}\n\n"
            "Return: position, strongest evidence, failure modes, and one actionable recommendation. "
            "Do not mention your model or provider identity."
        )

    @staticmethod
    def _scrub_identities(text: str, models: list[ModelSpec]) -> str:
        cleaned = text
        terms: set[str] = set()
        for model in models:
            terms.update({model.model, model.provider, model.family})
        for term in sorted((item for item in terms if len(item) >= 4), key=len, reverse=True):
            cleaned = re.sub(
                re.escape(term),
                "[model identity hidden]",
                cleaned,
                flags=re.IGNORECASE,
            )
        return cleaned

    def _run_moa(self, task: str, plan: Plan) -> CouncilResult:
        aggregator = plan.chairman
        advisors = [participant for participant in plan.participants if participant is not aggregator]
        if not advisors:
            return self._run_single(task, plan)
        prompts = [self._advisor_prompt(task, participant.role) for participant in advisors]
        answers, failures, calls = self._parallel(advisors, prompts, "advisor")
        if not answers:
            raise RuntimeError("All MoA advisors failed")
        references = "\n\n".join(
            f"REFERENCE {index + 1}:\n{text}" for index, (_, text) in enumerate(answers)
        )
        prompt = (
            f"Solve the task using the independent references below. Verify conflicts rather than "
            f"blindly averaging them.\n\nTASK:\n{task}\n\n{references}"
        )
        try:
            final = self.invoke(
                aggregator.model, prompt, "aggregator", aggregator.reasoning_effort
            )
        except Exception as exc:
            failures.append(
                f"aggregator ({aggregator.model.key}) failed: "
                f"{type(exc).__name__}: {exc}"
            )
            final = answers[0][1]
        return CouncilResult(plan.id, final, (), (), tuple(failures), calls + 1)

    def _run_council(
        self, task: str, plan: Plan, *, seed: int | None
    ) -> CouncilResult:
        chairman = plan.chairman
        advisors = [
            participant for participant in plan.participants if participant.role.startswith("advisor")
        ]
        if not advisors:
            raise ValueError("Council plan needs at least one advisor")
        advisor_prompts = [
            self._advisor_prompt(task, participant.role) for participant in advisors
        ]
        raw_answers, failures, advisor_calls = self._parallel(
            advisors, advisor_prompts, "advisor"
        )
        if not raw_answers:
            raise RuntimeError("All council advisors failed")

        rng = random.Random(seed)
        shuffled = list(raw_answers)
        rng.shuffle(shuffled)
        known_models = [participant.model for participant, _ in raw_answers]
        anonymous_answers = tuple(
            (
                chr(ord("A") + index),
                self._scrub_identities(text, known_models),
            )
            for index, (_, text) in enumerate(shuffled)
        )
        answer_block = "\n\n".join(
            f"Response {label}:\n{text}" for label, text in anonymous_answers
        )
        review_prompt = (
            "You are peer-reviewing anonymized council answers. Model identities are intentionally "
            "hidden. Judge only correctness, evidence, completeness, and actionable value.\n\n"
            f"TASK:\n{task}\n\n{answer_block}\n\n"
            "Return: strongest response, largest blind spot, key disagreement, and what all answers missed."
        )
        reviewers = advisors[: min(2, len(advisors))]
        reviews_raw, review_failures, review_calls = self._parallel(
            reviewers, [review_prompt] * len(reviewers), "reviewer"
        )
        failures.extend(review_failures)
        reviews = tuple(text for _, text in reviews_raw)
        review_block = "\n\n".join(
            f"Peer Review {index + 1}:\n{text}" for index, text in enumerate(reviews)
        ) or "No peer review succeeded; judge the anonymous answers directly."
        chairman_prompt = (
            "You are the chairman of a model council. Produce a decisive final answer without "
            "revealing hidden model identities. Preserve meaningful disagreement instead of forcing "
            "false consensus.\n\n"
            f"TASK:\n{task}\n\nANONYMOUS ANSWERS:\n{answer_block}\n\n"
            f"PEER REVIEWS:\n{review_block}\n\n"
            "Structure: Council agreement; important disagreements; blind spots; final recommendation; "
            "first concrete action."
        )
        try:
            final = self.invoke(
                chairman.model,
                chairman_prompt,
                "chairman",
                chairman.reasoning_effort,
            )
        except Exception as exc:
            failures.append(
                f"chairman ({chairman.model.key}) failed: "
                f"{type(exc).__name__}: {exc}"
            )
            label, text = anonymous_answers[0]
            final = f"Candidate {label}:\n{text}"
        return CouncilResult(
            plan.id,
            final,
            anonymous_answers,
            reviews,
            tuple(failures),
            advisor_calls + review_calls + 1,
        )
