"""Rule-based task profiling used before model selection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    kind: str
    complexity: int
    risk: int
    needs_tools: bool
    needs_freshness: bool
    benefits_from_diversity: bool


_ENGLISH_CODE = {
    "code", "coding", "bug", "debug", "test", "tests", "authentication",
    "api", "database", "refactor", "implementation",
}
_ENGLISH_RESEARCH = {
    "research", "search", "sources", "current", "latest", "news",
}
_ENGLISH_DECISION = {
    "choose", "decision", "compare", "tradeoff", "should", "architecture",
}
_ENGLISH_HIGH_RISK = {
    "production", "security", "authentication", "payment", "medical", "legal",
    "financial", "delete", "deploy",
}
_ENGLISH_COMPLEX = {
    "architecture", "multi-step", "migration", "distributed", "root cause",
    "review and fix",
}
_ENGLISH_TOOLS = _ENGLISH_CODE | {"file", "terminal", "browser", "website", "repository"}

_CHINESE_CODE = {"代码", "编程", "调试", "测试", "修复", "重构", "开发"}
_CHINESE_RESEARCH = {"研究", "搜索", "资料", "最新", "新闻", "调研"}
_CHINESE_DECISION = {"选择", "决策", "比较", "取舍", "方案", "架构"}
_CHINESE_HIGH_RISK = {"生产", "安全", "认证", "支付", "医疗", "法律", "金融", "删除", "部署"}
_CHINESE_COMPLEX = {"架构", "多步骤", "迁移", "分布式", "根因", "审查并修复", "重构", "系统设计"}
_CHINESE_TOOLS = _CHINESE_CODE | {"文件", "终端", "浏览器", "网站", "仓库"}

_BOUNDARY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(t) for t in sorted(
        _ENGLISH_CODE | _ENGLISH_RESEARCH | _ENGLISH_DECISION |
        _ENGLISH_HIGH_RISK | _ENGLISH_COMPLEX | _ENGLISH_TOOLS,
        key=len, reverse=True)) + r")(?![A-Za-z0-9_])"
)


def _matches(text: str, terms: set[str]) -> int:
    lower = text.lower()
    ascii_terms = {t.lower() for t in terms if t.isascii()}
    cjk_terms = {t.lower() for t in terms if not t.isascii()}
    english_hits = sum(1 for match in _BOUNDARY_PATTERN.findall(lower) if match in ascii_terms)
    cjk_hits = sum(1 for term in cjk_terms if term in lower)
    return english_hits + cjk_hits


def analyze_task(task: str) -> TaskProfile:
    """Classify a task into stable routing features without calling an LLM."""
    text = " ".join(str(task or "").lower().split())
    if not text:
        raise ValueError("task must not be empty")

    hits_code = _matches(text, _ENGLISH_CODE | _CHINESE_CODE)
    hits_research = _matches(text, _ENGLISH_RESEARCH | _CHINESE_RESEARCH)
    hits_decision = _matches(text, _ENGLISH_DECISION | _CHINESE_DECISION)
    counts = {"code": hits_code, "research": hits_research, "decision": hits_decision}
    kind = max(counts, key=lambda k: counts[k]) if any(counts.values()) else "general"

    risk_hits = _matches(text, _ENGLISH_HIGH_RISK | _CHINESE_HIGH_RISK)
    complex_hits = _matches(text, _ENGLISH_COMPLEX | _CHINESE_COMPLEX)
    complexity = min(5, 1 + min(2, len(text) // 180) + min(2, complex_hits * 2))
    risk = min(5, 1 + min(4, risk_hits * 2))
    needs_tools = bool(_matches(text, _ENGLISH_TOOLS | _CHINESE_TOOLS))
    needs_freshness = bool(_matches(text, _ENGLISH_RESEARCH | _CHINESE_RESEARCH))
    benefits = risk >= 4 or complexity >= 3 or kind == "decision"
    return TaskProfile(
        kind=kind,
        complexity=complexity,
        risk=risk,
        needs_tools=needs_tools,
        needs_freshness=needs_freshness,
        benefits_from_diversity=benefits,
    )
