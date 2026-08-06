"""Execution engine for single, MoA-style, and anonymous council plans."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .inventory import ModelSpec
from .recommender import Participant, Plan

Invoker = Callable[[ModelSpec, str, str, str], str]

_MAX_STAGE_TASK_CHARS = 6000
_MAX_REFERENCE_BLOCK_CHARS = 14000
_MAX_CHAIRMAN_ANSWER_CHARS = 9000
_MAX_CHAIRMAN_REVIEW_CHARS = 7000
_MAX_INVOKER_PROMPT_CHARS = 24000
_TRUNCATION_MARKER = "\n[content truncated for stage prompt budget]"


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
        suffix = "\n\nReturn a concise final answer in at most 1,200 words."
        prompt = self._clip(
            task,
            _MAX_INVOKER_PROMPT_CHARS - len(suffix),
        ) + suffix
        final = self.invoke(actor.model, prompt, "actor", actor.reasoning_effort)
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
                except Exception as exc:  # noqa: BLE001 - model invocation boundary
                    failures.append(
                        f"{role_prefix}-{index + 1} failed: {type(exc).__name__}"
                    )
        return [completed[index] for index in sorted(completed)], failures, len(participants)

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        value = str(text or "")
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars <= len(_TRUNCATION_MARKER):
            return _TRUNCATION_MARKER[:max_chars]
        keep = max_chars - len(_TRUNCATION_MARKER)
        return value[:keep] + _TRUNCATION_MARKER

    @classmethod
    def _bounded_block(
        cls, entries: list[tuple[str, str]], max_chars: int
    ) -> str:
        if not entries:
            return ""
        separators = 2 * (len(entries) - 1)
        heading_chars = sum(len(f"{heading}:\n") for heading, _ in entries)
        body_budget = max_chars - heading_chars - separators
        if body_budget < 0:
            headings = "\n\n".join(f"{heading}:\n" for heading, _ in entries)
            return cls._clip(headings, max_chars)
        per_entry, remainder = divmod(body_budget, len(entries))
        blocks = []
        for index, (heading, text) in enumerate(entries):
            allowance = per_entry + (1 if index < remainder else 0)
            blocks.append(f"{heading}:\n{cls._clip(text, allowance)}")
        return "\n\n".join(blocks)

    @classmethod
    def _advisor_prompt(cls, task: str, role: str) -> str:
        return (
            "You are an independent member of a model council. "
            f"Your assigned lens is {role}. Analyze independently and do not imitate consensus.\n\n"
            f"TASK:\n{cls._clip(task, _MAX_STAGE_TASK_CHARS)}\n\n"
            "Return at most 800 words: position, strongest evidence, failure modes, and one "
            "actionable recommendation. Do not mention your model or provider identity."
        )

    @staticmethod
    def _scrub_identities(text: str, models: list[ModelSpec]) -> str:
        cleaned = text
        terms: set[str] = set()
        family_aliases = {
            "anthropic": {"anthropic", "claude"},
            "openai": {"openai", "chatgpt", "codex", "gpt"},
            "google": {"google", "gemini"},
            "deepseek": {"deepseek"},
            "xai": {"xai", "grok"},
        }
        for model in models:
            terms.update({model.model, model.provider, model.family})
            terms.update(family_aliases.get(model.family.lower(), set()))
        for term in sorted((item for item in terms if len(item) >= 3), key=len, reverse=True):
            cleaned = re.sub(
                re.escape(term),
                "[model identity hidden]",
                cleaned,
                flags=re.IGNORECASE,
            )
        return cleaned

    def _run_moa(self, task: str, plan: Plan) -> CouncilResult:
        aggregator = plan.chairman
        advisors = [
            participant
            for participant in plan.participants
            if participant.role.startswith("advisor")
        ]
        if not advisors:
            return self._run_single(task, plan)
        prompts = [self._advisor_prompt(task, participant.role) for participant in advisors]
        answers, failures, calls = self._parallel(advisors, prompts, "advisor")
        if not answers:
            raise RuntimeError("All MoA advisors failed")
        references = self._bounded_block(
            [
                (f"REFERENCE {index + 1}", text)
                for index, (_, text) in enumerate(answers)
            ],
            _MAX_REFERENCE_BLOCK_CHARS,
        )
        prompt = (
            f"Solve the task using the independent references below. Verify conflicts rather than "
            f"blindly averaging them.\n\nTASK:\n"
            f"{self._clip(task, _MAX_STAGE_TASK_CHARS)}\n\n{references}\n\n"
            "Return a concise final answer in at most 1,200 words."
        )
        try:
            final = self.invoke(
                aggregator.model, prompt, "aggregator", aggregator.reasoning_effort
            )
        except Exception as exc:  # noqa: BLE001 - model invocation boundary
            failures.append(
                f"aggregator failed: {type(exc).__name__}"
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
        reviewers = advisors[: min(2, len(advisors))]
        review_prompts = []
        for reviewer in reviewers:
            reviewer_answers = [
                (participant, text)
                for participant, text in raw_answers
                if participant.model.key != reviewer.model.key
            ]
            reviewer_rng = random.Random(rng.random())
            reviewer_rng.shuffle(reviewer_answers)
            reviewer_models = [participant.model for participant, _ in reviewer_answers]
            reviewer_anonymous = tuple(
                (
                    chr(ord("A") + index),
                    self._scrub_identities(text, reviewer_models),
                )
                for index, (_, text) in enumerate(reviewer_answers)
            )
            reviewer_block = self._bounded_block(
                [(f"Response {label}", text) for label, text in reviewer_anonymous],
                _MAX_REFERENCE_BLOCK_CHARS,
            ) or "No other candidate answer succeeded."
            review_prompts.append(
                "You are peer-reviewing anonymized council answers. Model identities are intentionally "
                "hidden. Judge only correctness, evidence, completeness, and actionable value.\n\n"
                f"TASK:\n{self._clip(task, _MAX_STAGE_TASK_CHARS)}\n\n"
                f"{reviewer_block}\n\n"
                "Return at most 800 words: strongest response, largest blind spot, key disagreement, "
                "and what all answers missed."
            )
        reviews_raw, review_failures, review_calls = self._parallel(
            reviewers, review_prompts, "reviewer"
        )
        failures.extend(review_failures)
        reviews = tuple(
            self._scrub_identities(text, known_models) for _, text in reviews_raw
        )
        chairman_answer_block = self._bounded_block(
            [(f"Response {label}", text) for label, text in anonymous_answers],
            _MAX_CHAIRMAN_ANSWER_CHARS,
        )
        review_block = self._bounded_block(
            [(f"Peer Review {index + 1}", text) for index, text in enumerate(reviews)],
            _MAX_CHAIRMAN_REVIEW_CHARS,
        ) or "No peer review succeeded; judge the anonymous answers directly."
        chairman_prompt = (
            "You are the chairman of a model council. Produce a decisive final answer without "
            "revealing hidden model identities. Preserve meaningful disagreement instead of forcing "
            "false consensus.\n\n"
            f"TASK:\n{self._clip(task, _MAX_STAGE_TASK_CHARS)}\n\n"
            f"ANONYMOUS ANSWERS:\n{chairman_answer_block}\n\n"
            f"PEER REVIEWS:\n{review_block}\n\n"
            "Structure in at most 1,200 words: Council agreement; important disagreements; "
            "blind spots; final recommendation; first concrete action."
        )
        try:
            final = self.invoke(
                chairman.model,
                chairman_prompt,
                "chairman",
                chairman.reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001 - model invocation boundary
            failures.append(
                f"chairman failed: {type(exc).__name__}"
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
