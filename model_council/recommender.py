"""Generate Pareto-style multi-model plans from task and model metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import TaskProfile
from .inventory import ModelSpec


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

    @property
    def chairman(self) -> Participant:
        for participant in reversed(self.participants):
            if participant.role in {"chairman", "aggregator", "actor"}:
                return participant
        return self.participants[-1]


_BUDGET_NAMES = ("luna", "haiku", "mini", "flash")
_PREMIUM_NAMES = ("fable", "opus", "sol-pro", "sol", "pro")


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
    for model in ranked:
        if model.family not in used_families:
            selected.append(model)
            used_families.add(model.family)
            if len(selected) == limit:
                return selected
    for model in ranked:
        if model not in selected:
            selected.append(model)
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
    usable = [model for model in models if model.healthy is not False]
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
        if model != primary
    ]
    if not references:
        references = [
            model for model in _rank(balanced_usable, profile, "advisor") if model != primary
        ]
    reference = references[0] if references else primary
    balanced_risks = ["增加一次参考模型调用", *execution_risks]
    if reference.family == primary.family:
        balanced_risks.append("当前没有第二个可用模型家族，独立性有限")
    balanced = Plan(
        id="balanced",
        label="均衡/MoA",
        mode="moa",
        participants=(
            Participant("advisor", reference, _effort(profile)),
            Participant("aggregator", primary, _effort(profile, premium=True)),
        ),
        estimated_calls=2,
        max_calls=3,
        strengths=("独立 Advisor 与 Aggregator", "模型调用复用 Hermes Provider 配置"),
        risks=tuple(balanced_risks),
    )

    advisors = _diverse_selection(_rank(usable, profile, "advisor"), min(3, len(usable)))
    chairman_model = _rank(usable, profile, "chairman")[0]
    reviewer_count = min(2, len(advisors))
    calls = len(advisors) + reviewer_count + 1
    quality_participants = [
        Participant(f"advisor-{index}", model, _effort(profile, premium=True))
        for index, model in enumerate(advisors, start=1)
    ]
    quality_participants.append(
        Participant("chairman", chairman_model, _effort(profile, premium=True))
    )
    quality_risks = ["调用次数与延迟最高", "匿名互评会消耗额外上下文", *execution_risks]
    if len({model.family for model in advisors}) < 2:
        quality_risks.append("模型家族多样性不足")
    quality = Plan(
        id="quality",
        label="质量优先/Council",
        mode="council",
        participants=tuple(quality_participants),
        estimated_calls=calls,
        max_calls=9,
        strengths=("独立生成", "匿名互评", "Chairman 仲裁"),
        risks=tuple(quality_risks),
    )
    return [fast, balanced, quality]
