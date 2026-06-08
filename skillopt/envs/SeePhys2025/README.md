# SeePhys2025 Environment

This folder adds a full SkillOpt environment adapter for SeePhys-style physics vision QA.

The optimization target is:
- input: `question` + `images`
- label: `answer`

During rollout, the target model is instructed to answer directly in plain text. A separate Qwen judge compares the full response against the gold answer(s) and returns JSON scoring fields.

## 1) Files and Roles

### `adapter.py`

`SeePhys2025Adapter` is the bridge between SkillOpt trainer and this benchmark.

Main responsibilities:
- initialize and configure loader (`setup`)
- build train/eval env batches (`build_train_env`, `build_eval_env`)
- run rollout (`rollout`) for target model inference
- run reflection (`reflect`) so optimizer can propose edits to the skill

In short: this is the entry point trainer talks to.

### `dataloader.py`

`SeePhys2025DataLoader` handles dataset read + normalization + split.

Main responsibilities:
- load HF dataset from disk (`load_from_disk`)
- avoid Pillow dependency by casting images with `decode=False`
- materialize image bytes to real files in cache dir (`_materialize_images`)
- normalize row into common SkillOpt item schema (`_normalize_item`)
- split data to train/val/test with robust ratio logic (`_split_items`)
- sample train/eval batches deterministically by seed

Why materialize images:
- rollout requires stable file paths
- avoids runtime image decode dependency issues

### `rollout.py`

Executes one item (or batch) against target backend and writes prediction artifacts.

Main responsibilities:
- build system/user prompts and multimodal messages (`_build_messages`)
- attach images as data-URI for chat backends
- call target backend (`chat_target_messages`) or exec backend path
- keep the target response in plain text so the judge can read the full reasoning and answer
- evaluate each response and write per-item artifacts under `predictions/`

Artifacts saved per item typically include:
- `target_system_prompt.txt`
- `target_user_prompt.txt`
- `conversation.json`

If using exec backend path, a local workspace is prepared and a per-item `skill.md` is generated in that workspace.

### `evaluator.py`

Computes metrics from prediction text and gold labels.

Main responsibilities:
- send the full model response plus the gold answer(s) to a Qwen judge
- parse JSON output with `hard`, `soft`, and `reason`
- return structured result fields (`hard`, `soft`, `predicted_answer`, `gold_answers`)

Current behavior:
- `hard` and `soft` intentionally use the same 0/1 verdict
- fallback scoring compares normalized raw response text with gold answer(s) if the judge call fails

### `__init__.py`

Package marker for environment module import.

### `README.md`

This file.

### `skills/initial.md`

Seed skill used when you want the trainer to start from a non-empty prompt.

Main purpose:
- teach the model to ground answers in the images
- keep the final answer concise in plain text
- reduce early-step wandering when the skill is still untrained

## 2) Config File

Default config:
- `configs/SeePhys2025/default.yaml`

Inheritance chain:
- `configs/SeePhys2025/default.yaml` sets `_base_: ['../_base_/default.yaml']`
- `configs/_base_/default.yaml` provides the shared SkillOpt defaults
- the SeePhys2025 file then overrides the environment-specific values on top of that base

How to read the effective runtime config:
- the trainer loads the structured YAML, merges the base file first, then applies the SeePhys2025 overrides
- the merged structured config is flattened into the CLI/runtime keys used by `scripts/train.py`
- structured overrides use `section.key=value` form, for example `model.target=...` or `evaluation.use_gate=true`

Key sections:
- `env`: dataset path, split ratio, timeout
- `train`: epochs, batch size, seed
- `optimizer`: learning-rate-like edit budget and schedulers
- `evaluation`: gate selection/test behavior
- `model`: backend + model deployments + Qwen chat connection

Current default is `qwen_chat` for both optimizer and target.

Dataset split behavior in this env:
- the loader currently loads one dataset from `env.data_path`, shuffles it deterministically with `env.split_seed`, and splits it into train/val/test by `env.split_ratio`
- `env.split_mode` is accepted in config, but this SeePhys2025 loader path uses the ratio split logic in code
- `env.limit` caps the total number of items before the ratio split happens
- `train.train_size` is not an independent per-split selector here; the trainer infers train size from the loaded train split, and if you set `train.train_size > 0` it must match the actual train split size exactly

If you want explicit counts, use this rule:
- choose `env.limit = train_count + val_count + test_count`
- set `env.split_ratio` to the same proportion as those counts, for example `4:1:5`
- if you need an exact train count, make sure the resulting ratio and limit produce that count, or change the loader code to support direct count-based splitting

Important base-config note:
- the base config still defines generic model and execution defaults such as `model.reasoning_effort`, `model.rewrite_reasoning_effort`, `model.rewrite_max_completion_tokens`, and the Codex/Claude/Azure execution fields
- SeePhys2025 does not use those defaults unless you switch the backend, but they are part of the merged config and can be overridden in either file

## 3) Output Skill Files: Which File Is Which?

After training, SkillOpt writes:

- `outputs/<run>/best_skill.md`
	- best validated skill selected by gate over training steps
	- this is the main skill file to reuse/deploy

- `outputs/<run>/skills/skill_v0000.md`, `skill_v0001.md`, ...
	- snapshot after each step
	- useful for debugging evolution trajectory

- `outputs/<run>/steps/step_xxxx/...`
	- rollout, patches, and per-step records

About `skill.md` naming:
- trainer's canonical final file is `best_skill.md`
- per-step files use `skills/skill_v*.md`
- a plain `skill.md` appears in per-item exec workspaces when exec backend path is used

## 4) How to Run Training (and what each command does)

Use repo venv in this project:

```bash
cd /media/hung/DATA/SkillOpt
./improve_skill/bin/python scripts/train.py --help
```

### A. Smoke training (quick sanity run)

```bash
cd /media/hung/DATA/SkillOpt
./improve_skill/bin/python scripts/train.py \
	--config configs/SeePhys2025/default.yaml \
	--skill_init skillopt/envs/SeePhys2025/skills/initial.md \
	--num_epochs 1 \
	--batch_size 4 \
	--limit 40 \
	--workers 4 \
	--out_root outputs/seephys_smoke
```

What this does:
- runs a short loop to verify end-to-end data -> rollout -> reflect -> update -> evaluate
- limits dataset for fast feedback
- writes initial and evolved skill files into `outputs/seephys_smoke`

Expected skill artifacts:
- `outputs/seephys_smoke/best_skill.md`
- `outputs/seephys_smoke/skills/skill_v*.md`

### B. Full training (produce stronger best skill)

```bash
cd /media/hung/DATA/SkillOpt
./improve_skill/bin/python scripts/train.py \
	--config configs/SeePhys2025/default.yaml \
	--skill_init skillopt/envs/SeePhys2025/skills/initial.md \
	--out_root outputs/seephys_full
```

What this does:
- uses default epochs/batch/scheduler in config
- performs gate-based selection each step
- updates best skill when candidate passes validation
- starts from the seeded SeePhys skill so the first rollout already has image-grounded guidance

Main output to use later:
- `outputs/seephys_full/best_skill.md`

If you explicitly want to start from a blank skill instead, remove `--skill_init skillopt/envs/SeePhys2025/skills/initial.md`.

If you want to keep the seeded prompt but use a different output directory:

```bash
cd /media/hung/DATA/SkillOpt
./improve_skill/bin/python scripts/train.py \
	--config configs/SeePhys2025/default.yaml \
	--skill_init skillopt/envs/SeePhys2025/skills/initial.md \
	--out_root outputs/seephys_custom
```

What this does:
- keeps the first-step prompt seeded with the SeePhys guidance
- makes the initial rollout less brittle when the benchmark is still cold-start

### C. Resume a stopped run

```bash
cd /media/hung/DATA/SkillOpt
./improve_skill/bin/python scripts/train.py \
	--config configs/SeePhys2025/default.yaml \
	--out_root outputs/seephys_full
```

What this does:
- if `runtime_state.json` exists in the same `out_root`, training resumes from next unfinished step
- avoids redoing finished steps

### D. Force output filename `skill.md` (optional convenience)

If you specifically want a root-level `skill.md` file for downstream tooling:

```bash
cd /media/hung/DATA/SkillOpt
cp outputs/seephys_full/best_skill.md outputs/seephys_full/skill.md
```

What this does:
- keeps canonical SkillOpt outputs unchanged
- creates an alias filename for tools expecting `skill.md`

## 5) Required Runtime Conditions

For default config, ensure Qwen chat endpoint is reachable:
- `model.qwen_chat_base_url` (default: `http://localhost:8000/v1`)
- `model.optimizer` and `model.target` deployment names exist on that endpoint

If the endpoint/model names differ, override via CLI:

```bash
cd /media/hung/DATA/SkillOpt
./improve_skill/bin/python scripts/train.py \
	--config configs/SeePhys2025/default.yaml \
	--qwen_chat_base_url http://<host>:<port>/v1 \
	--optimizer_model <your_model_name> \
	--target_model <your_model_name> \
	--out_root outputs/seephys_custom
```

## 6) Evaluation Metric in This Env

The metric is defined in code, not by a loose heuristic.

Per sample, the rollout writes the raw model response and then calls `evaluate(prediction_text, gold, question)` from `skillopt/envs/SeePhys2025/evaluator.py`.

The judge input is a JSON payload with:
- `question`
- `gold_answers`
- `model_response`

The Qwen judge must return strict JSON in this shape:

```json
{"hard": 0|1, "soft": 0|1, "reason": "..."}
```

Exact scoring behavior:
- `hard` is converted to `0` or `1` and used as the main correctness signal.
- `soft` is forced to the same value as `hard`, so it does not represent an independent metric in this env.
- If the judge returns invalid JSON or the request fails, the fallback is normalized exact-match against the gold answer(s).
- Normalization removes whitespace, `$`, and commas, then lowercases the text before comparison.
- A blank prediction only counts as correct in fallback mode when the gold answer is also blank.

Aggregation behavior:
- `skillopt.utils.compute_score` reports the mean of `hard` and the mean of `soft` across the evaluated batch.
- The trainer stores both values, but selection/gate decisions use `hard` as the candidate score.

Current interpretation in this env:
- `hard` means "acceptable/correct" versus "unacceptable/wrong".
- `soft` is intentionally mirrored from `hard`.
- `selection_hard` is the value used by the gate when deciding whether a candidate skill replaces the current one.

## 7) Skill Update Flow

Aggregation and gate behavior:
- `skillopt.utils.compute_score` returns the batch mean of `hard` and the batch mean of `soft`
- the trainer records both, but selection and gate decisions use `hard` as the candidate score
- `selection_hard` is the value stored for the selection split and is what the gate compares when deciding whether to accept a candidate skill

### Reflect and update flow

Reflection happens after rollout and before the selection gate:
1. The rollout results are split into failures and successes by `hard`.
2. `run_minibatch_reflect()` groups each side into minibatches of size `optimizer.minibatch_size`.
3. For failures, it calls `run_error_analyst_minibatch()`; for successes, it calls `run_success_analyst_minibatch()` unless `optimizer.failure_only=true`.
4. Each analyst call receives the current skill plus the formatted trajectories from that minibatch.
5. The returned patch is applied to produce a candidate skill, which is then re-evaluated on the selection split.
6. If `evaluation.use_gate=true`, the trainer accepts the candidate only when the selection `hard` score passes the gate.

What reflect uses as input:
- the current skill document
- the rollout trajectories from the minibatch
- the target system prompt and target user prompt saved for each trajectory
- the raw conversation from `conversation.json`
- hidden reference text when the trajectory has one
- previous-step buffer context and meta-skill context when provided by the trainer

What reflect does not receive directly:
- it does not get raw image tensors as model inputs
- it does not reload image files as a separate filesystem context; the images are attached to the optimizer message as multimodal `image_url` parts built from the saved rollout image paths
- it does not re-run the multimodal target prompt; it works from the saved rollout artifacts and trajectory text

Which prompt is used for reflect:
- failure minibatches use `skillopt/prompts/analyst_error.md`
- success minibatches use `skillopt/prompts/analyst_success.md`
- if `optimizer.skill_update_mode` switches to a rewrite-style mode, the reflector tries the matching rewrite prompt first, such as `analyst_error_rewrite.md` or `analyst_success_rewrite.md`
- for full rewrite modes, the reflector uses the matching `_full_rewrite` prompt variant when available

How the prompt is structured:
- the system prompt comes from the prompt file above
- the user prompt starts with `## Current Skill`
- then it adds the minibatch budget or full-rewrite instruction
- then it adds `## Previous Steps in This Epoch` when step-buffer context exists
- then it appends meta-skill context if present
- finally it appends either `## Failed Trajectories (...)` or `## Successful Trajectories (...)` with all formatted trajectory text

Slow update path:
- when `optimizer.use_slow_update=true`, the trainer also runs an end-of-epoch longitudinal update
- that branch samples `optimizer.slow_update_samples` train items, compares previous and current skill rollouts, and generates `slow_update_content`
- if content is produced, it is force-injected into both `current_skill` and `best_skill` unconditionally
- this slow-update content is not gated by the step-level selection score

The model updates in the standard SkillOpt loop:
- `hard` means "acceptable/correct" versus "unacceptable/wrong"
- `soft` is intentionally mirrored from `hard`
- `selection_hard` is the value used by the gate when deciding whether a candidate skill replaces the current one
| Flag | Value |
## 8) Current Flags and Values
| `model.backend` | `qwen_chat` |
| `model.optimizer_backend` | `qwen_chat` |
| `model.target_backend` | `qwen_chat` |
| `model.optimizer` | `Qwen/Qwen3.6-27B` |
| `model.target` | `Qwen/Qwen3.6-27B` |
| `model.qwen_chat_base_url` | `http://localhost:8000/v1` |
| `model.qwen_chat_timeout_seconds` | `300` |
| `model.qwen_chat_max_tokens` | `8000` |
| `model.qwen_chat_enable_thinking` | `true` |

Optimizer/evaluation:

| `model.reasoning_effort` | `medium` |
| `model.rewrite_reasoning_effort` | `""` |
| `model.rewrite_max_completion_tokens` | `64000` |
| Flag | Value |
| --- | --- |
| `optimizer.learning_rate` | `4` |
| `optimizer.lr_scheduler` | `cosine` |
| `optimizer.skill_update_mode` | `patch` |
| `optimizer.use_slow_update` | `true` |
| `optimizer.use_meta_skill` | `true` |
| `evaluation.use_gate` | `true` |
| `evaluation.sel_env_num` | `16` |
| `evaluation.test_env_num` | `32` |
| `optimizer.min_learning_rate` | `2` |
| `evaluation.eval_test` | `true` |
| `optimizer.lr_control_mode` | `fixed` |

| `optimizer.slow_update_samples` | `20` |
| `optimizer.longitudinal_pair_policy` | `mixed` |
Train/env:

| Flag | Value |
| --- | --- |
| `train.num_epochs` | `4` |
| `train.batch_size` | `32` |
| `train.seed` | `42` |
| `gradient.minibatch_size` | `8` |
| `gradient.analyst_workers` | `16` |
| `env.data_path` | `SeePhys_data` |
| `env.split_mode` | `ratio` |
| `env.split_ratio` | `4:1:5` |
| `train.train_size` | `0` |
| `env.split_seed` | `42` |
| `train.accumulation` | `1` |
| `env.limit` | `400` |
| `env.exec_timeout` | `180` |
| `gradient.merge_batch_size` | `8` |

| `gradient.max_analyst_rounds` | `3` |
| `gradient.failure_only` | `false` |
| `env.name` | `SeePhys2025` |
When you override these from the CLI, the script uses the same structured config keys and passes the resulting flat values into the adapter and trainer.

## 9) Refer to Global Project README

- `README.md` at repository root
| `env.limit` | `400` |

Base defaults that still exist in the merged config but are usually not used in this env unless you switch the backend:
- `model.azure_openai_*` fields are empty by default
- `model.codex_exec_*` fields come from the base config and control Codex-based execution paths
- `model.claude_code_exec_*` fields also come from the base config and stay at their shared defaults

This env README focuses only on SeePhys2025-specific behavior and outputs.
