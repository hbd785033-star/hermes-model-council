"""Execution engine for single, MoA-style, and anonymous council plans."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .decision import (
    CouncilResult,
    DecisionProcess,
    DecisionRecord,
    DecisionStatus,
)
from .inventory import ModelSpec
from .lenses import resolve_advisor_lens
from .recommender import (
    Participant,
    Plan,
    canonical_model_identity,
    has_independent_candidate,
)

Invoker = Callable[[ModelSpec, str, str, str], str]

_MAX_STAGE_TASK_CHARS = 6000
_MAX_REFERENCE_BLOCK_CHARS = 14000
_MAX_CHAIRMAN_ANSWER_CHARS = 9000
_MAX_CHAIRMAN_REVIEW_CHARS = 7000
_MAX_INVOKER_PROMPT_CHARS = 24000
_TRUNCATION_MARKER = "\n[content truncated for stage prompt budget]"


def _failure_code(exc: Exception) -> str:
    """Return a useful failure reason without exposing provider or model identity."""
    text = str(exc or "").casefold()
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return "timeout"
    if "429" in text or "rate limit" in text or "rate_limit" in text:
        return "rate_limited"
    if any(term in text for term in ("authentication", "unauthorized", "401", "403")):
        return "authentication"
    if "empty response" in text:
        return "empty_response"
    return "invocation_error"


class _DecisionExecutionFailure(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        observed_calls: int,
        warnings: tuple[str, ...],
        actual_process: DecisionProcess,
    ):
        super().__init__(reason)
        self.reason = reason
        self.observed_calls = observed_calls
        self.warnings = warnings
        self.actual_process = actual_process


class CouncilRunner:
    def __init__(self, invoke: Invoker, max_workers: int = 4):
        self.invoke = invoke
        self.max_workers = max(1, int(max_workers))

    @staticmethod
    def _eligible_reviewers(
        advisors: list[Participant],
        candidate_participants: list[Participant],
    ) -> list[Participant]:
        if len(candidate_participants) < 2:
            return []
        candidate_models = tuple(candidate.model for candidate in candidate_participants)
        return [
            reviewer
            for reviewer in advisors
            if has_independent_candidate(reviewer.model, candidate_models)
        ][:2]

    @classmethod
    def _required_calls(cls, plan: Plan) -> int:
        if plan.mode == "single":
            return 1
        if plan.mode == "moa":
            return len(plan.participants)
        if plan.mode == "council":
            advisor_count = sum(
                participant.role.startswith("advisor")
                for participant in plan.participants
            )
            advisors = [
                participant
                for participant in plan.participants
                if participant.role.startswith("advisor")
            ]
            return advisor_count + len(cls._eligible_reviewers(advisors, advisors)) + 1
        return 0

    def run(self, task: str, plan: Plan, *, seed: int | None = None) -> DecisionRecord:
        if not str(task or "").strip():
            raise ValueError("task must not be empty")
        required_calls = self._required_calls(plan)
        has_advisor = any(
            participant.role.startswith("advisor")
            for participant in plan.participants
        )
        invalid_topology_overestimate = (
            plan.mode == "council"
            and not has_advisor
            and plan.estimated_calls > plan.max_calls
        )
        if required_calls > plan.max_calls or invalid_topology_overestimate:
            enforced_calls = (
                plan.estimated_calls if invalid_topology_overestimate else required_calls
            )
            raise ValueError(
                f"Plan exceeds call budget: {enforced_calls} > {plan.max_calls}"
            )
        try:
            if plan.mode == "single":
                result = self._run_single(task, plan)
            elif plan.mode == "moa":
                result = self._run_moa(task, plan)
            elif plan.mode == "council":
                result = self._run_council(task, plan, seed=seed)
            else:
                raise ValueError(f"Unsupported plan mode: {plan.mode}")
        except _DecisionExecutionFailure as exc:
            return DecisionRecord(
                status=DecisionStatus.FAILED,
                decision=None,
                process=exc.actual_process,
                preset=plan.id,
                models_consulted=tuple(
                    canonical_model_identity(participant.model)
                    for participant in plan.participants[:exc.observed_calls]
                ),
                configured_call_ceiling=plan.max_calls,
                topology_required_calls=required_calls,
                observed_calls=exc.observed_calls,
                degraded_reasons=(exc.reason,),
                warnings=exc.warnings,
            )
        if result.actual_process is None:
            raise ValueError("execution result is missing actual process evidence")
        return DecisionRecord(
            status=(
                DecisionStatus.DEGRADED
                if result.degraded
                else DecisionStatus.COMPLETED
            ),
            decision=result.final,
            process=result.actual_process,
            preset=plan.id,
            models_consulted=tuple(
                dict.fromkeys(
                    canonical_model_identity(participant.model)
                    for participant in plan.participants
                )
            ),
            configured_call_ceiling=plan.max_calls,
            topology_required_calls=required_calls,
            observed_calls=result.call_count,
            fallback_used=result.fallback_source is not None,
            fallback_reason=self._fallback_reason(result),
            degraded_reasons=(
                (result.degradation_reason,)
                if result.degradation_reason is not None
                else ()
            ),
            warnings=result.failures,
            process_evidence=result,
        )


    @staticmethod
    def _fallback_reason(result: CouncilResult) -> str | None:
        if result.fallback_source is None:
            return None
        if result.degradation_reason == "chairman_failed":
            return "chairman_failed_candidate_fallback"
        if result.degradation_reason == "aggregator_failed":
            return "aggregator_failed_advisor_fallback"
        return "process_fallback"

    def _run_single(
        self,
        task: str,
        plan: Plan,
        *,
        forced_reason: str | None = None,
    ) -> CouncilResult:
        actor = plan.chairman
        suffix = "\n\nReturn a concise final answer in at most 1,200 words."
        prompt = self._clip(
            task,
            _MAX_INVOKER_PROMPT_CHARS - len(suffix),
        ) + suffix
        try:
            final = str(
                self.invoke(actor.model, prompt, "actor", actor.reasoning_effort) or ""
            ).strip()
            if not final:
                raise RuntimeError("empty response")
        except (RuntimeError, TimeoutError) as exc:
            code = _failure_code(exc)
            raise _DecisionExecutionFailure(
                "single_invocation_failed",
                observed_calls=1,
                warnings=(f"actor failed: {code}",),
                actual_process=DecisionProcess.SINGLE,
            ) from exc
        task_truncated = len(str(task)) > _MAX_INVOKER_PROMPT_CHARS - len(suffix)
        degradation_reason = forced_reason or plan.degradation_reason
        if degradation_reason is None and task_truncated:
            degradation_reason = "task_truncated"
        return CouncilResult(
            plan.id,
            final,
            (),
            (),
            (),
            1,
            plan.degraded or forced_reason is not None or task_truncated,
            degradation_reason,
            1,
            0.0,
            None,
            task_truncated,
            DecisionProcess.SINGLE,
        )

    def _parallel(
        self, participants: list[Participant], prompts: list[str], role_prefix: str
    ) -> tuple[list[tuple[Participant, str]], list[str], int]:
        if not participants:
            return [], [], 0
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
                except (RuntimeError, TimeoutError) as exc:
                    failures.append(
                        f"{role_prefix}-{index + 1} failed: {_failure_code(exc)}"
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
        lens = resolve_advisor_lens(role)
        return (
            "You are an independent member of a model council. "
            f"Decision lens: {lens.id}\n"
            f"{lens.instruction}\n"
            "Apply this lens silently. Do not mention the lens name or identifier in your answer. "
            "Analyze independently and do not imitate consensus.\n\n"
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
        for term in sorted(
            (item for item in terms if len(item) >= 3), key=len, reverse=True
        ):
            tokens = re.findall(r"[a-z0-9]+", term.casefold())
            if not tokens:
                continue
            pattern = (
                r"(?<![a-z0-9])"
                + r"[\W_]*".join(re.escape(token) for token in tokens)
                + r"(?![a-z0-9])"
            )
            cleaned = re.sub(
                pattern,
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
            return self._run_single(task, plan, forced_reason="no_advisors")
        prompts = [self._advisor_prompt(task, participant.role) for participant in advisors]
        answers, failures, calls = self._parallel(advisors, prompts, "advisor")
        if not answers:
            raise _DecisionExecutionFailure(
                "all_moa_advisors_failed",
                observed_calls=calls,
                warnings=tuple(failures),
                actual_process=DecisionProcess.CUSTOM_MOA,
            )
        known_models = [participant.model for participant in plan.participants]
        answers = [
            (participant, self._scrub_identities(text, known_models))
            for participant, text in answers
        ]
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
        aggregator_failed = False
        fallback_source: str | None = None
        try:
            final = str(
                self.invoke(
                    aggregator.model, prompt, "aggregator", aggregator.reasoning_effort
                )
                or ""
            ).strip()
            if not final:
                raise RuntimeError("empty response")
        except (RuntimeError, TimeoutError) as exc:
            aggregator_failed = True
            failures.append(
                f"aggregator failed: {_failure_code(exc)}"
            )
            final = answers[0][1]
            successful_participant = answers[0][0]
            successful_index = next(
                index
                for index, participant in enumerate(advisors)
                if participant is successful_participant
            )
            fallback_source = f"advisor-{successful_index + 1}"
        final = self._scrub_identities(final, known_models)
        task_truncated = len(str(task)) > _MAX_STAGE_TASK_CHARS
        degraded = (
            plan.degraded
            or bool(failures)
            or len(answers) < len(advisors)
            or task_truncated
        )
        degradation_reason = plan.degradation_reason
        if aggregator_failed:
            degradation_reason = "aggregator_failed"
        elif failures:
            degradation_reason = "participant_failure"
        elif degradation_reason is None and task_truncated:
            degradation_reason = "task_truncated"
        return CouncilResult(
            plan.id,
            final,
            (),
            (),
            tuple(failures),
            calls + 1,
            degraded,
            degradation_reason,
            len(answers),
            0.0,
            fallback_source,
            task_truncated,
            DecisionProcess.CUSTOM_MOA,
        )

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
            raise _DecisionExecutionFailure(
                "all_council_advisors_failed",
                observed_calls=advisor_calls,
                warnings=tuple(failures),
                actual_process=DecisionProcess.CUSTOM_COUNCIL,
            )

        rng = random.Random(seed)
        shuffled = list(raw_answers)
        rng.shuffle(shuffled)
        # Scrub every participant identity, including chairman and failed advisors.
        known_models = [participant.model for participant in plan.participants]
        candidates = tuple(
            (
                f"candidate-{index:02d}",
                participant,
                self._scrub_identities(text, known_models),
            )
            for index, (participant, text) in enumerate(shuffled, start=1)
        )
        anonymous_answers = tuple(
            (candidate_id, text) for candidate_id, _, text in candidates
        )
        reviewers = self._eligible_reviewers(
            advisors,
            [participant for _, participant, _ in candidates],
        )
        review_prompts = []
        for reviewer in reviewers:
            reviewer_answers = [
                (candidate_id, participant, text)
                for candidate_id, participant, text in candidates
                if canonical_model_identity(participant.model)
                != canonical_model_identity(reviewer.model)
            ]
            reviewer_rng = random.Random(rng.random())
            reviewer_rng.shuffle(reviewer_answers)
            reviewer_block = self._bounded_block(
                [
                    (f"Response {candidate_id}", text)
                    for candidate_id, _, text in reviewer_answers
                ],
                _MAX_REFERENCE_BLOCK_CHARS,
            )
            review_prompts.append(
                "You are peer-reviewing anonymized council answers. Model identities are intentionally "
                "hidden. Judge only correctness, evidence, completeness, and actionable value.\n\n"
                f"TASK:\n{self._clip(task, _MAX_STAGE_TASK_CHARS)}\n\n"
                f"{reviewer_block}\n\n"
                "Return at most 800 words: strongest response, largest blind spot, key disagreement, "
                "and what all answers missed. Refer to candidates only by their exact candidate IDs."
            )
        reviews_raw, review_failures, review_calls = self._parallel(
            reviewers, review_prompts, "reviewer"
        )
        failures.extend(review_failures)
        reviews = tuple(
            self._scrub_identities(text, known_models) for _, text in reviews_raw
        )
        review_coverage = len(reviews) / len(reviewers) if reviewers else 0.0
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
        fallback_source: str | None = None
        chairman_failed = False
        try:
            final = str(
                self.invoke(
                    chairman.model,
                    chairman_prompt,
                    "chairman",
                    chairman.reasoning_effort,
                )
                or ""
            ).strip()
            if not final:
                raise RuntimeError("empty response")
            final = self._scrub_identities(final, known_models)
        except (RuntimeError, TimeoutError) as exc:
            chairman_failed = True
            failures.append(
                f"chairman failed: {_failure_code(exc)}"
            )
            label, text = anonymous_answers[0]
            final = f"Candidate {label}:\n{text}"
            fallback_source = label
        candidate_count = len(anonymous_answers)
        advisor_failed = len(raw_answers) < len(advisors)
        task_truncated = len(str(task)) > _MAX_STAGE_TASK_CHARS
        degraded = (
            plan.degraded
            or candidate_count < 2
            or review_coverage < 1.0
            or advisor_failed
            or bool(review_failures)
            or chairman_failed
            or task_truncated
        )
        degradation_reason: str | None = plan.degradation_reason
        if chairman_failed:
            degradation_reason = "chairman_failed"
        elif advisor_failed:
            degradation_reason = "participant_failure"
        elif candidate_count < 2:
            degradation_reason = "insufficient_candidates"
        elif review_coverage < 1.0:
            degradation_reason = "peer_review_incomplete"
        elif degradation_reason is None and task_truncated:
            degradation_reason = "task_truncated"
        return CouncilResult(
            plan.id,
            final,
            anonymous_answers,
            reviews,
            tuple(failures),
            advisor_calls + review_calls + 1,
            degraded,
            degradation_reason,
            candidate_count,
            review_coverage,
            fallback_source,
            task_truncated,
            DecisionProcess.CUSTOM_COUNCIL,
        )
