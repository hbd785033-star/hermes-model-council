"""Generate Pareto-style multi-model plans from task and model metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .analysis import TaskProfile
from .inventory import ModelSpec
from .lenses import select_decision_lenses


@dataclass(frozen=True)
class Participant:
    role: str
    model: ModelSpec
    reasoning_effort: str


@dataclass(frozen=True)
class Plan:
    id: str
    label: str
    mode: str
    participants: tuple[Participant, ...]
    estimated_calls: int
    max_calls: int
    strengths: tuple[str, ...]
    risks: tuple[str, ...]
    degraded: bool = False
    degradation_reason: str | None = None

    @property
    def chairman(self) -> Participant:
        for participant in reversed(self.participants):
            if participant.role in {"chairman", "aggregator", "actor"}:
                return participant
        return self.participants[-1]


_BUDGET_NAMES = ("luna", "haiku", "mini", "flash")
_PREMIUM_NAMES = ("fable", "opus", "sol-pro", "sol", "pro")


def _normalized_model_name(name: str) -> str:
    """Conservatively normalize cosmetic aliases in display model names."""
    value = re.sub(r"[^a-z0-9]+", "-", str(name or "").casefold()).strip("-")
    return value


def canonical_model_identity(model: ModelSpec) -> str:
    """Return a conservative display-name identity used for diversity boundaries.

    Inventory currently has no authoritative underlying-model ID, so this only
    collapses case, whitespace, punctuation and an explicit provider/family
    prefix. It deliberately does not claim semantic alias resolution.
    """
    value = str(model.model or "").casefold().strip()
    for separator in (":", "/"):
        if separator in value:
            prefix, remainder = value.split(separator, 1)
            provider_tokens = {
                _normalized_model_name(model.provider),
                _normalized_model_name(model.family),
            }
            if _normalized_model_name(prefix) in provider_tokens:
                value = remainder
                break
    return _normalized_model_name(value)


def has_independent_candidate(
    reviewer: ModelSpec,
    candidates: list[ModelSpec] | tuple[ModelSpec, ...],
) -> bool:
    """Return whether a reviewer has at least one non-self candidate."""
    reviewer_identity = canonical_model_identity(reviewer)
    return any(
        canonical_model_identity(candidate) != reviewer_identity
        for candidate in candidates
    )


def _tier_score(model: ModelSpec) -> float:
    name = model.model.lower()
    score = 2.0
    if any(term in name for term in _PREMIUM_NAMES):
        score += 2.5
    if any(term in name for term in _BUDGET_NAMES):
        score -= 0.8
    if model.reasoning:
        score += 0.5
    if model.is_current:
        score += 0.2
    if model.healthy is True:
        score += 0.3
    return score


def _task_score(model: ModelSpec, profile: TaskProfile, role: str) -> float:
    score = _tier_score(model)
    family = model.family
    name = model.model.lower()
    if profile.kind == "code":
        score += 3.0 if family == "openai" or "codex" in name else 0.0
        score += 1.5 if family == "anthropic" else 0.0
    elif profile.kind == "research":
        score += 2.5 if family in {"google", "anthropic"} else 0.5
    elif profile.kind == "decision":
        score += 2.5 if family == "anthropic" else 1.0
    if role in {"chairman", "aggregator"}:
        score += 2.0 if family == "anthropic" else 0.8
    if role == "fast":
        score += 2.0 if model.fast else 0.0
        score += 2.0 if any(term in name for term in _BUDGET_NAMES) else 0.0
        score -= 1.0 if any(term in name for term in ("fable", "opus", "pro")) else 0.0
    return score


def _rank(models: list[ModelSpec], profile: TaskProfile, role: str) -> list[ModelSpec]:
    return sorted(models, key=lambda item: _task_score(item, profile, role), reverse=True)


def _diverse_selection(ranked: list[ModelSpec], limit: int) -> list[ModelSpec]:
    selected: list[ModelSpec] = []
    used_families: set[str] = set()
    used_models: set[str] = set()
    for model in ranked:
        identity = canonical_model_identity(model)
        if identity in used_models:
            continue
        if model.family not in used_families:
            selected.append(model)
            used_families.add(model.family)
            used_models.add(identity)
            if len(selected) == limit:
                return selected
    for model in ranked:
        identity = canonical_model_identity(model)
        if model not in selected and identity not in used_models:
            selected.append(model)
            used_models.add(identity)
            if len(selected) == limit:
                break
    return selected


def _effort(profile: TaskProfile, premium: bool = False) -> str:
    if premium or profile.risk >= 4 or profile.complexity >= 4:
        return "high"
    if profile.complexity >= 3:
        return "medium"
    return "low"


def recommend_plans(profile: TaskProfile, models: list[ModelSpec]) -> list[Plan]:
    """Return fast, balanced, and quality plans; never select known-broken models."""
    usable_by_key: dict[str, ModelSpec] = {}
    for model in models:
        if model.healthy is not False:
            usable_by_key.setdefault(model.key, model)
    usable = list(usable_by_key.values())
    if not usable:
        raise ValueError("No healthy or unverified models are available")

    execution_risks = ["自定义 run 子会话隔离，不继承会话上下文"]
    if profile.needs_tools:
        execution_risks.append(
            "自定义 run 禁用工具；工具任务应使用 Hermes 原生 MoA Preset"
        )

    fast_model = _rank(usable, profile, "fast")[0]
    fast = Plan(
        id="fast",
        label="快速/低调用",
        mode="single",
        participants=(Participant("actor", fast_model, "low"),),
        estimated_calls=1,
        max_calls=1,
        strengths=("最低延迟", "只调用一个模型"),
        risks=("没有独立交叉检查", *execution_risks),
    )

    if len(usable) == 1:
        def degraded(plan_id: str, label: str) -> Plan:
            return Plan(
                id=plan_id,
                label=label,
                mode="single",
                participants=(
                    Participant("actor", fast_model, _effort(profile, premium=True)),
                ),
                estimated_calls=1,
                max_calls=1,
                strengths=("保留已验证模型的可用结果", "不制造虚假多模型共识"),
                risks=("仅有一个已验证健康模型，本方案已降级为单模型", *execution_risks),
                degraded=True,
                degradation_reason="single_model_inventory",
            )

        return [
            fast,
            degraded("balanced", "均衡/已降级单模型"),
            degraded("quality", "质量优先/已降级单模型"),
        ]

    balanced_usable = [model for model in usable if "fable" not in model.model.lower()] or usable
    primary = _rank(balanced_usable, profile, "aggregator")[0]
    references = [
        model
        for model in _diverse_selection(_rank(balanced_usable, profile, "advisor"), 3)
        if (
            model != primary
            and canonical_model_identity(model) != canonical_model_identity(primary)
        )
    ]
    if not references:
        references = [
            model
            for model in _rank(balanced_usable, profile, "advisor")
            if (
                model != primary
                and canonical_model_identity(model) != canonical_model_identity(primary)
            )
        ]
    reference = references[0] if references else primary
    balanced_advisors: list[Participant] = []
    seen_families: set[str] = set()
    for ref in references:
        if ref.family not in seen_families:
            balanced_advisors.append(
                Participant(f"advisor-{len(balanced_advisors)+1}", ref, _effort(profile))
            )
            seen_families.add(ref.family)
    if not balanced_advisors:
        balanced_advisors = [Participant("advisor", primary, _effort(profile))]
    balanced_advisors = [
        Participant(f"advisor-{lens.id}", participant.model, participant.reasoning_effort)
        for participant, lens in zip(
            balanced_advisors,
            select_decision_lenses(len(balanced_advisors)),
        )
    ]
    balanced_participants = (
        *balanced_advisors,
        Participant("aggregator", primary, _effort(profile, premium=True)),
    )
    balanced_risks = ["增加参考模型调用", *execution_risks]
    if reference.family == primary.family:
        balanced_risks.append("当前没有第二个可用模型家族，独立性有限")
    if canonical_model_identity(reference) == canonical_model_identity(primary):
        balanced_risks.append("Advisor 与 Aggregator 使用同名模型，独立性有限")
    balanced = Plan(
        id="balanced",
        label="均衡/MoA",
        mode="moa",
        participants=balanced_participants,
        estimated_calls=len(balanced_participants),
        max_calls=max(3, len(balanced_participants) + 1),
        strengths=("独立 Advisor 与 Aggregator", "模型调用复用 Hermes Provider 配置"),
        risks=tuple(balanced_risks),
    )

    distinct_model_names = {canonical_model_identity(model) for model in usable}
    advisor_limit = min(3, max(1, len(distinct_model_names) - 1))
    advisors = _diverse_selection(
        _rank(usable, profile, "advisor"), advisor_limit
    )
    advisor_keys = {model.key for model in advisors}
    advisor_names = {canonical_model_identity(model) for model in advisors}
    chairman_pool = [
        model
        for model in usable
        if model.key not in advisor_keys
        and canonical_model_identity(model) not in advisor_names
    ]
    chairman_pool = chairman_pool or [m for m in usable if m.key not in advisor_keys] or usable
    chairman_model = _rank(chairman_pool, profile, "chairman")[0]
    reviewer_count = len([
        model for model in advisors
        if has_independent_candidate(model, tuple(advisors))
    ][:2]) if len(advisors) >= 2 else 0
    calls = len(advisors) + reviewer_count + 1
    quality_participants = [
        Participant(f"advisor-{lens.id}", model, _effort(profile, premium=True))
        for model, lens in zip(advisors, select_decision_lenses(len(advisors)))
    ]
    quality_participants.append(
        Participant("chairman", chairman_model, _effort(profile, premium=True))
    )
    quality_risks = ["调用次数与延迟最高", *execution_risks]
    if reviewer_count:
        quality_risks.append("匿名互评会消耗额外上下文")
    else:
        quality_risks.append("成功候选不足两个时跳过互评并结构化披露降级")
    if chairman_model.key in advisor_keys:
        quality_risks.append("没有独立的Chairman模型，已降级与Advisor同源")
    if canonical_model_identity(chairman_model) in advisor_names:
        quality_risks.append("没有独立的Chairman模型名，已降级与Advisor同名")
    if len({model.family for model in advisors}) < 2:
        quality_risks.append("模型家族多样性不足")
    if reviewer_count:
        quality_risks.append("评审者也参与候选生成，但不评审自己的候选答案")
    quality_strengths = ["独立生成", "Chairman 仲裁"]
    if reviewer_count:
        quality_strengths.insert(1, "匿名互评")
    quality = Plan(
        id="quality",
        label="质量优先/Council",
        mode="council",
        participants=tuple(quality_participants),
        estimated_calls=calls,
        max_calls=9,
        strengths=tuple(quality_strengths),
        risks=tuple(quality_risks),
    )
    return [fast, balanced, quality]
