---
name: model-council
description: Use when choosing or coordinating Hermes models.
version: 0.1.0
author: Hermes Model Council contributors
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [model-routing, council, moa, multi-agent, review]
    related_skills: [hermes-agent, multi-model-orchestration]
---

# Model Council

## Overview

Use the D-drive Model Council executor to discover the user's actually configured Hermes models, classify the task, present three comparable plans, and execute only the plan the user approves. It complements Hermes native MoA with anonymous peer review and Chairman synthesis.

Project launcher:

```text
python "C:\Users\DHB\AppData\Local\hermes\skills\model-council\scripts\model_council.py"
```

The launcher uses `MODEL_COUNCIL_HOME` when set; otherwise it uses `D:\Projects\hermes-model-council`.

## When to Use

Use this skill when the user:

- asks which model or model combination fits a task;
- asks for a second opinion, council, jury, debate, critic, reviewer, or judge;
- has a high-risk decision, security review, architecture choice, or consequential analysis;
- wants quality/cost/latency alternatives before model calls;
- asks to run Hermes MoA with models that are actually available now.

Do not use it for an ordinary low-risk task already being handled well by one model unless the user explicitly requests comparison. Never assume more models means better.

## Required Workflow

### 1. Profile and probe

Run:

```text
python "C:\Users\DHB\AppData\Local\hermes\skills\model-council\scripts\model_council.py" recommend "<task>" --probe --json
```

Completion criterion: JSON contains `task_profile`, `plans`, live `health_diagnostics`, `probe_call_count`, and `probe_cache_hit_count`; only verified models appear as available after probing. Successful health results are cached for 15 minutes and failures for 2 minutes to avoid repeated Hermes session overhead without masking transient recovery. Use `--refresh-probe` only when a fresh check is required.

For sensitive input, do not probe or broadcast the full task until the user approves sending it to every Provider shown. First run `inventory --json`, describe the candidate Providers, and ask which Providers may receive the data.

### 2. Present exactly three choices

Always show:

1. `fast` — one-model baseline;
2. `balanced` — Advisor → Aggregator;
3. `quality` — independent answers → stable-ID anonymous peer review → Chairman; if fewer than two candidates can be produced, show the explicit degraded path instead of claiming peer review.

For each choice include:

- Provider/model and role;
- topology and estimated calls;
- main strength;
- cost/latency/privacy trade-off;
- known failed or excluded Providers;
- fallback behavior.

Do not invent prices. If price metadata is unavailable, say `cost: unknown; calls: N`.

### 3. Obtain approval

Use `clarify` to let the user select `fast`, `balanced`, or `quality`, unless the user already explicitly selected one in the current turn. Do not execute a multi-model plan merely because the task is complex.

Completion criterion: the user has selected one named plan and has accepted any multi-Provider privacy warning.

### 4. Execute the selected plan

After approval, run:

```text
python "C:\Users\DHB\AppData\Local\hermes\skills\model-council\scripts\model_council.py" run "<task>" --plan <fast|balanced|quality> --yes --json
```

- `fast`: one isolated Hermes model call.
- `balanced`: one independent advisor followed by one aggregator.
- `quality`: parallel independent advisors, randomized anonymous responses, up to two peer reviews, and Chairman synthesis.

Report `probe_call_count`, `probe_cache_hit_count`, `execution_call_count`, `total_call_count`, every failed participant, and whether the result degraded. Never hide fallback or substitute a Provider silently. Hermes sessions use source `model-council`, so token usage can be inspected with `hermes insights --source model-council`.

### 5. Use native MoA for tool work

The custom Council disables tools in its child sessions. For a task whose execution requires web, files, terminal, or other Hermes tools, install/update native presets only after approval:

```text
python "C:\Users\DHB\AppData\Local\hermes\skills\model-council\scripts\model_council.py" install-presets --yes --json
```

Then use the generated Hermes preset:

```text
/moa:model-council-balanced
```

Use `model-council-quality` only when the extra calls have a clear quality role. Configuration writes must have a timestamped backup and pass `hermes config check`.

## Council Protocol

1. **Independent generation** — assign different lenses; do not show answers to other advisors.
2. **Anonymization** — assign each candidate one stable opaque label, randomize display order without relabeling it, and remove model and Provider identifiers.
3. **Peer review** — require at least two successful candidates; rank correctness, evidence, completeness, blind spots, and disagreement without self-review.
4. **Chairman** — state agreement, preserve meaningful disagreement, identify blind spots, make one recommendation, and give the first concrete action.
5. **Disclosure** — return call count, failures, degraded paths, and limitations.

The programmatic Evidence Layer is opt-in. Do not claim a normal CLI Council run passed an Evidence Gate unless a trusted external verifier actually produced the artifacts and its verifier ID was explicitly allowed by the gate policy.

## Privacy Rules

- Never place credentials in prompts, logs, project files, or summaries.
- A sensitive task defaults to one Provider until the user permits multi-Provider transmission.
- Do not persist task prompts or model outputs unless the user explicitly asks.
- Do not treat a configured Provider as healthy: live probe it.
- After live probing, unknown models are unavailable for that run.

## Common Pitfalls

1. **Skipping the single-model baseline.** Always include `fast`.
2. **Using an expensive model as a ceremonial voter.** Every extra model needs a distinct role.
3. **Claiming native tools in custom Council.** Custom children are tool-free; use native MoA for tools.
4. **Rotating through unverified models after failure.** Failed representative candidates disable that Provider for the run.
5. **Showing brands during review.** Reviewers see anonymous labels only.
6. **Executing before approval.** Recommendation and execution are separate steps.
7. **Claiming peer review with one candidate.** Skip the empty reviewer stage and disclose `degraded=true`, `degradation_reason=insufficient_candidates`, and `review_coverage=0`.

## Verification Checklist

- [ ] Inventory came from live Hermes configuration
- [ ] Health status came from live probes when execution is planned
- [ ] Fast, balanced, and quality options were all shown
- [ ] Models have differentiated roles
- [ ] Call budget and privacy warning were visible
- [ ] User selected a plan before execution
- [ ] Failures and fallback were disclosed
- [ ] Candidate IDs stayed stable across Reviewer and Chairman prompts
- [ ] Peer review ran only with at least two successful candidates
- [ ] Tool-requiring work used native MoA rather than custom tool-free Council
