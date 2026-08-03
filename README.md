# Hermes Model Council

一个面向 Hermes Agent 的任务感知模型推荐与协作层。它读取当前 Hermes 实际配置的模型，先给出可比较的方案，再在用户批准后执行。

## 能力

- 从 Hermes 原生 inventory 动态发现 Provider、模型和当前模型。
- 规则式任务画像：类型、复杂度、风险、工具需求、时效性和多样性收益。
- 三档 Pareto 方案：
  - `fast`：单模型基线，1 次调用。
  - `balanced`：Advisor → Aggregator，通常 2 次调用。
  - `quality`：并行独立回答 → 匿名互评 → Chairman，默认最多 9 次调用。
- 实时健康探测；live probe 后只有实际通过的型号才会被推荐。
- 调用预算、并发上限、超时、失败披露和 Provider 降级。
- 生成 Hermes 原生 `model-council-balanced` / `model-council-quality` MoA Preset。
- 默认不记录任务正文或模型输出；仅由 Hermes 自身管理其子会话。

## 设计边界

Model Council 不替代 Hermes Provider、OAuth、安全、工具和会话体系。所有模型调用仍通过 `hermes chat`，不会复制 API Key，也不在项目文件中保存凭据。

自定义 Council 执行器的子调用默认禁用工具，用于独立推理和匿名评审。需要工具的任务应安装并使用 Hermes 原生 MoA Preset。

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

`--probe` 会进行最小、无工具的 `HEALTH_OK` 调用。探测会产生少量模型调用。

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

写配置前自动创建时间戳备份。安装后可以在 Hermes 中使用：

```text
/moa:model-council-balanced
/moa:model-council-quality
```

## 隐私与安全

- 不读取或输出 Token、API Key、OAuth 内容。
- 子进程使用参数数组，`shell=False`，避免命令注入。
- 错误信息进行常见凭据格式脱敏。
- Council 评审隐藏 Provider 和模型型号，并随机化候选顺序。
- 同一敏感任务发送给多个云 Provider 前，必须由用户选择方案。
- 已知失败的 Provider 不会静默换另一个未验证型号。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 当前限制

- 成本展示目前以调用次数和模型档位为主；Hermes inventory 没有稳定返回价格时，不伪造美元估算。
- 自定义 Council 通过命令行传递提示词，单次提示限制为 24,000 字符。
- live probe 只在当前命令中生效，默认不持久化健康数据。
- 模型能力评分是可解释规则，尚未使用历史反馈训练路由器。
