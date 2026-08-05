"""Rule-based task profiling used before model selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    kind: str
    complexity: int
    risk: int
    needs_tools: bool
    needs_freshness: bool
    benefits_from_diversity: bool


_CODE_TERMS = {
    "code", "coding", "bug", "debug", "test", "tests", "authentication",
    "api", "database", "refactor", "implementation", "代码", "编程", "调试",
    "测试", "修复", "重构", "开发",
}
_RESEARCH_TERMS = {
    "research", "search", "sources", "current", "latest", "news", "研究",
    "搜索", "资料", "最新", "新闻", "调研",
}
_DECISION_TERMS = {
    "choose", "decision", "compare", "tradeoff", "should", "architecture",
    "选择", "决策", "比较", "取舍", "方案", "架构",
}
_HIGH_RISK_TERMS = {
    "production", "security", "authentication", "payment", "medical", "legal",
    "financial", "delete", "deploy", "生产", "安全", "认证", "支付", "医疗",
    "法律", "金融", "删除", "部署",
}
_COMPLEX_TERMS = {
    "architecture", "multi-step", "migration", "distributed", "root cause",
    "review and fix", "架构", "多步骤", "迁移", "分布式", "根因", "审查并修复",
    "重构", "系统设计",
}
_TOOL_TERMS = _CODE_TERMS | {
    "file", "terminal", "browser", "website", "repository", "文件", "终端",
    "浏览器", "网站", "仓库",
}


def _matches(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def analyze_task(task: str) -> TaskProfile:
    """Classify a task into stable routing features without calling an LLM."""
    text = " ".join(str(task or "").lower().split())
    if not text:
        raise ValueError("task must not be empty")

    counts = {
        "code": _matches(text, _CODE_TERMS),
        "research": _matches(text, _RESEARCH_TERMS),
        "decision": _matches(text, _DECISION_TERMS),
    }
    kind = max(counts, key=lambda name: counts[name]) if any(counts.values()) else "general"
    risk_hits = _matches(text, _HIGH_RISK_TERMS)
    complex_hits = _matches(text, _COMPLEX_TERMS)
    complexity = min(5, 1 + min(2, len(text) // 180) + min(2, complex_hits * 2))
    risk = min(5, 1 + min(4, risk_hits * 2))
    needs_tools = bool(_matches(text, _TOOL_TERMS))
    needs_freshness = bool(_matches(text, _RESEARCH_TERMS))
    benefits = risk >= 4 or complexity >= 3 or kind == "decision"
    return TaskProfile(
        kind=kind,
        complexity=complexity,
        risk=risk,
        needs_tools=needs_tools,
        needs_freshness=needs_freshness,
        benefits_from_diversity=benefits,
    )
