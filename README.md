# Hermes Model Council

一个面向 Hermes Agent 的任务感知模型推荐与协作层。它读取当前 Hermes 实际配置的模型，先给出可比较的方案，再在用户批准后执行。

## 能力

- 从 Hermes 原生 inventory 动态发现 Provider、模型和当前模型。
- 规则式任务画像：类型、复杂度、风险、工具需求、时效性和多样性收益。
- 三档 Pareto 方案：
  - `fast`：单模型基线，1 次调用。
  - `balanced`：一至多个不同模型的 Advisor → Aggregator，通常 2～4 次调用；以输出的 `estimated_calls` 为准。
  - `quality`：并行独立回答 → 稳定匿名候选 ID 互评 → Chairman，默认最多 9 次调用；不足两个成功候选时跳过空互评并结构化披露降级。
- 实时健康探测；只有清理后的完整响应严格等于 `HEALTH_OK` 才视为健康，live probe 后只有实际通过的型号才会被推荐。
- 调用预算、阶段提示预算、并发上限、超时、失败披露和 Provider 降级。
- 生成 Hermes 原生 `model-council-balanced` / `model-council-quality` MoA Preset。
- 默认不记录任务正文或模型输出；仅由 Hermes 自身管理其子会话。

## 设计边界

Model Council 不替代 Hermes Provider、OAuth、安全、工具和会话体系。所有模型调用仍通过 `hermes chat`，不会复制 API Key，也不在项目文件中保存凭据。

自定义 Council 执行器的子调用默认禁用工具并隔离会话，用于独立推理和匿名评审。需要工具或当前会话上下文的任务应安装并使用 Hermes 原生 MoA Preset。

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

JSON 输出会分别报告 `probe_call_count`、`probe_cache_hit_count`、`execution_call_count` 和 `total_call_count`。旧字段 `call_count` 保留为执行阶段调用数。这些字段统计 Model Council 发起的 Hermes 子会话；Provider 内部对 429 等错误的 HTTP 重试不由 Hermes CLI 暴露，因此不计入。Council 结果还包含 `degraded`、`degradation_reason`、`candidate_count`、`review_coverage`、`fallback_source` 和 `task_truncated`，调用方无需解析自由文本即可识别降级或回退。

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
- Council 候选、peer review 和失败诊断默认都隐藏 Provider、模型型号及常见模型家族别名。候选只随机化一次并分配稳定的 `candidate-NN` ID，Reviewer 子集与 Chairman 全程保持同一映射；失败诊断仅显示席位和安全原因码（如 `timeout`、`rate_limited`）。
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
