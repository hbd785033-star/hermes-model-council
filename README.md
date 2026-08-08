# Hermes Model Council

一个面向 Hermes Agent 的任务感知模型推荐与协作层。它读取当前 Hermes 实际配置的模型，先给出可比较的方案，再在用户批准后执行。

## 能力

- 从 Hermes 原生 inventory 动态发现 Provider、模型和当前模型。
- 规则式任务画像：类型、复杂度、风险、工具需求、时效性和多样性收益。
- 三档 Pareto 方案：
  - `fast`：单模型基线，1 次调用。
  - `balanced`：一至多个不同模型的 Advisor → Aggregator，通常 2～4 次调用；以输出的 `estimated_calls` 为准。
  - `quality`：至少两个成功候选时执行并行独立回答 → 稳定匿名 ID 互评 → Chairman，默认最多 9 次调用；候选不足时跳过空评审并显式降级。
- 实时健康探测；只有清理后的完整响应严格等于 `HEALTH_OK` 才视为健康，live probe 后只有实际通过的型号才会被推荐。
- 调用预算、阶段提示预算、并发上限、超时、失败披露和 Provider 降级。
- 生成 Hermes 原生 `model-council-balanced` / `model-council-quality` MoA Preset。
- 默认不记录任务正文或模型输出；仅由 Hermes 自身管理其子会话。

## 设计边界

Model Council 不替代 Hermes Provider、OAuth、安全、工具和会话体系。所有模型调用仍通过 `hermes chat`，不会复制 API Key，也不在项目文件中保存凭据。

自定义 Council 执行器的子调用默认禁用工具并隔离会话，用于独立推理和匿名评审。需要工具或当前会话上下文的任务应安装并使用 Hermes 原生 MoA Preset。

## Evidence Layer（程序化接口）

`model_council.evidence` 提供结构化 `Claim`、`EvidenceArtifact`、`EvidenceBundle` 和确定性的 `EvidenceGate`。门禁只判断外部验证器提交的状态，不让 LLM 自报 confidence 或“我已验证”直接通过关键声明。

```python
from pathlib import Path

from model_council import (
    Claim,
    ClaimImportance,
    CommandVerifier,
    EvidenceBundle,
    EvidenceGate,
)

claim = Claim("tests", "项目测试通过", ClaimImportance.REQUIRED)
verifier = CommandVerifier(
    root=Path("D:/Projects/example"),
    allowed_executables=("python", "python.exe"),
)
artifact = verifier.verify(
    "unit-tests",
    claim.id,
    ("python", "-m", "unittest", "discover", "-s", "tests"),
)
verdict = EvidenceGate(trusted_verifiers=(artifact.verifier,)).evaluate(
    EvidenceBundle(claims=(claim,), artifacts=(artifact,))
)
```

`EvidenceGate` 默认不信任任何 verifier；只有调用方明确加入 `trusted_verifiers` 的外部验证器才能满足关键声明，模型自报的 `verified` 状态会被忽略并出现在 `untrusted_evidence_ids`。`CommandVerifier` 固定使用参数数组和 `shell=False`，限制工作目录必须位于配置的根目录内，并只允许预先配置的可执行文件；allowlist 名称会在初始化时解析为可信绝对路径，同名但路径不一致的可执行文件会被拒绝。它是执行边界，不是容器沙箱；命令必须来自可信应用配置，绝不能直接来自模型输出或用户任务正文。当前 CLI 尚不自动生成 Claim 或运行验证命令，避免在没有明确权限和任务成功标准时扩大工具权限。

`CitationVerifier` 只验证调用方预先允许的 HTTPS host 上是否存在指定原文摘录。它拒绝 userinfo、query、fragment、IP 字面量、非 443 端口和解析到非公网地址的 host；默认抓取器禁用自动重定向，并限制响应大小和文本内容类型。它证明的是“该来源包含这段原文”，不是自动证明原文推导出的结论正确；引用蕴含关系仍应作为独立 Eval。

`EntailmentPolicy` 汇总受信 evaluator 对 Claim–Citation 关系的 `supported`、`contradicted` 或 `insufficient` 判断。不同 evaluator 同时给出支持和反驳时，结果降为 `insufficient` 并标记 disagreement；模型自评和未加入 allowlist 的 assessment 被忽略。该策略默认只 advisory；即使请求 hard gate，也必须先满足校准样本数、Pearson 相关性和严重误放行率阈值。默认阈值为至少 100 个 golden samples、Pearson ≥ 0.7、严重误放行率 ≤ 5%。

## Outcome / Telemetry Store（程序化接口）

`TelemetryStore` 使用本地 SQLite 保存按事件划分的任务画像和结果元数据，为离线 Evals 与未来 Router 提供聚合数据。schema 只包含事件 ID、UTC 时间、任务类型、复杂度/风险、计划/角色、Provider/模型/家族、结果、验证分数、延迟、调用量、Token 数、安全失败码、枚举式用户反馈和策略版本；不包含原始 Prompt、模型输出、凭据或完整工具轨迹。

```python
from pathlib import Path
from model_council import FeedbackKind, OutcomeEvent, OutcomeKind, TelemetryStore

store = TelemetryStore(Path("D:/Projects/hermes-model-council-data/telemetry.db"))
store.record(OutcomeEvent(
    event_id="run-001", occurred_at="2026-08-08T12:00:00+00:00",
    task_kind="security_review", complexity=4, risk=5,
    plan_id="quality", role="advisor", provider="provider-a",
    model="model-a", family="family-a", outcome=OutcomeKind.SUCCESS,
    evaluator_score=0.9, latency_ms=1200, execution_calls=3,
    total_tokens=800, failure_code=None, feedback=FeedbackKind.POSITIVE,
    policy_version="router-v1",
))
summary = store.summarize(task_kind="security_review")
```

标识字段只允许受限 identifier 字符，避免把自由文本伪装成 `task_kind` 等字段写入；反馈只允许 `positive/negative/none`。默认保留 90 天，过期事件在同一次写入事务中清理或拒绝落盘。每个 SQLite 操作显式关闭连接，兼容 Windows 临时目录与文件锁语义，并提供 `integrity_check()`。当前 schema version 为 2；v1 数据库会原子迁移并保留历史事件，将 Token 列升级为可空。绩效聚合包含样本量、成功/失败/未知结果、正负反馈、平均 evaluator score、延迟、调用数和已知 Token 均值；未知 Token 不参与平均。当前 Store 不会自动修改 Router 权重。

真实 Council 执行默认不写 telemetry；使用以下显式 opt-in：

```bash
python -m model_council run "任务" --plan quality --yes --telemetry
python -m model_council run "任务" --plan quality --yes --telemetry --telemetry-path "D:/ModelCouncilData/outcomes.db"
```

`TelemetryInvoker` 为每次 Advisor/Reviewer/Aggregator/Chairman 调用记录一条事件；原 Prompt、模型输出和异常正文不会进入数据库。Hermes CLI 没有暴露单次调用 Token 数时存为 SQL `NULL`，不伪造为 0。Telemetry 初始化或写入失败采用 best-effort 降级，不改变原模型调用的返回值、异常或 Council fallback；`--telemetry-path` 只有与 `--telemetry` 同时使用才会创建数据库。

执行成功后，输出会包含 `telemetry_run_id`。最终任务绩效必须由用户、外部 Eval 或回放系统另行提交，不能由 Chairman 自动宣布：

```bash
python -m model_council record-outcome \
  --run-id "run-20260808T120000000000Z" \
  --outcome success \
  --evaluator-score 0.9 \
  --feedback positive \
  --telemetry-path "D:/ModelCouncilData/outcomes.db"
```

`record-outcome` 从该 run 的 per-call 事件反查任务类型、复杂度/风险、plan、累计调用数和延迟，然后写入独立 `role=run` 事件。相同 run ID 的最终 outcome 不可覆盖；`summarize_runs()` 只聚合 `role=run`，不会把“模型 API 调用成功”误当成“任务答案正确”。

积累足够 run outcomes 后可生成只读离线绩效报告：

```bash
python -m model_council performance-report \
  --task-kind security_review \
  --baseline-plan fast \
  --minimum-samples 30 \
  --telemetry-path "D:/ModelCouncilData/outcomes.db" \
  --json
```

报告按 plan 展示已知 outcome 数、成功率与 Wilson 95% 区间、正反馈率、外部 Eval 均分、延迟/调用/已知 Token 均值，并分别计算 success、score 和 latency regret。未知 outcome 不进入成功率分母；低于最小样本量的 plan 不计算 regret。报告不生成单一综合分或 `recommended_plan`，也不会写数据库或修改 Router。

## 安装

项目默认位于：

```text
D:\Projects\hermes-model-council
```

可选安装 CLI：

```bash
python -m pip install -e .
```

不安装也可直接运行：

```bash
python -m model_council --help
```

Hermes Skill 源文件位于：

```text
skill/model-council/
```

本机安装后的 Skill 启动器默认查找 D 盘项目。其他位置可设置：

```bash
export MODEL_COUNCIL_HOME='D:/your/path/hermes-model-council'
```

## 使用

### 1. 查看 Hermes 当前模型

```bash
python -m model_council inventory
```

### 2. 生成三个方案

```bash
python -m model_council recommend '审查生产认证代码并给出修复方案' --probe
```

`--probe` 会进行最小、无工具的 `HEALTH_OK` 调用。成功结果默认缓存 15 分钟，失败结果只缓存 2 分钟，避免短任务重复承担 Hermes 子会话的固定上下文成本，同时不会长时间放大瞬时故障；使用 `--refresh-probe` 可强制重新探测。

JSON 输出会分别报告 `probe_call_count`、`probe_cache_hit_count`、`execution_call_count` 和 `total_call_count`。旧字段 `call_count` 保留为执行阶段调用数。执行结果还包含 `degraded`、`degradation_reason`、`candidate_count`、`review_coverage`、`fallback_source` 和 `task_truncated`，避免把降级路径伪装成完整 Council。这些字段统计 Model Council 发起的 Hermes 子会话；Provider 内部对 429 等错误的 HTTP 重试不由 Hermes CLI 暴露，因此不计入。

健康探测有意串行执行，避免并发 Hermes CLI 子会话产生状态竞争；15 分钟成功缓存用于降低重复探测延迟。

### 3. 用户确认后执行

```bash
python -m model_council run '任务正文' --plan fast --yes
python -m model_council run '任务正文' --plan balanced --yes
python -m model_council run '任务正文' --plan quality --yes
```

没有 `--yes` 时拒绝执行，避免误触发昂贵 Council。

### 4. 安装 Hermes 原生 MoA Preset

```bash
python -m model_council install-presets --yes
```

写配置前自动创建时间戳备份，写入后执行 `hermes config check`；检查失败时自动恢复备份。安装后可以在 Hermes 中使用：

```text
/moa:model-council-balanced
/moa:model-council-quality
```

## 隐私与安全

- 不读取或输出 Token、API Key、OAuth 内容。
- 子进程使用参数数组，`shell=False`，避免命令注入。
- 错误信息进行常见凭据格式脱敏。
- Council 候选、peer review 和失败诊断默认都隐藏 Provider、模型型号及常见模型家族别名，并随机化候选顺序；每个候选的匿名 ID 在 Reviewer 与 Chairman 阶段保持稳定，失败诊断仅显示席位和安全原因码（如 `timeout`、`rate_limited`）。
- 任务正文通过命令行 `hermes chat -q` 的 argv 参数传入，在本机进程列表中可见；敏感内容建议先确认 Provider 集合，再运行含正文的推荐/执行命令。
- 同一敏感任务发送给多个云 Provider 前，必须由用户选择方案。
- 已知失败的 Provider 不会静默换另一个未验证型号。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 当前限制

- 成本展示目前以调用次数和模型档位为主；Hermes inventory 没有稳定返回价格时，不伪造美元估算。
- 自定义 Council 通过命令行传递提示词，单次提示限制为 24,000 字符；跨阶段材料会按预算裁剪并显式标记。
- 自定义子调用的输出长度通过角色提示软约束（Advisor/Reviewer 800 词，Actor/Aggregator/Chairman 1,200 词）；Hermes CLI 暂无 Provider 级输出 Token 参数。原生 MoA Preset 使用硬性 `max_tokens=4096`，参考输出为 600/900 Token。
- live probe 结果只持久化模型键、健康布尔值和检查时间，不保存任务、模型输出或错误详情；缓存文件默认位于用户缓存目录的 `hermes-model-council/health-cache.json`，也可用 `MODEL_COUNCIL_CACHE_DIR` 指定目录。缓存不可写时本次运行会跳过持久化，不会阻断推荐。
- Hermes 子会话使用独立来源标签 `model-council`，可通过 `hermes insights --source model-council` 查看实际 Token 使用。
- 模型能力评分是可解释规则，尚未使用历史反馈训练路由器。
