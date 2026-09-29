<a id="top"></a>

## PDEU-Bench: Benchmarking Personalized Planning of LLM Agents under Explicit and Implicit User Preferences

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/rabbitrose/PDEU-Bench/blob/main/LICENSE)

> **PDEU-Bench** is a benchmark for evaluating whether LLM agents can *plan* in a way that faithfully follows individual user preferences while calling tools. Unlike tool-use benchmarks that ask "can the agent finish the task?", PDEU-Bench isolates and measures the **planning ability** of an agent as it maintains and revises a per-user *plan list* throughout an interaction.

> 🌟 Find PDEU-Bench useful? Star the repo to follow updates and show your support!

## 🖼️ Framework Overview

PDEU-Bench evaluates personalized planning along **three sequential capabilities**, each scored on preference adherence and plan quality:

1. **First Plan (Planning Design)** — Given a task and a user, the agent produces an initial plan list. The headline metric is *First Plan Acc*.
2. **Instantiation (Planning Instantiation)** — Each time the agent instantiates a JSON object against an API schema, we measure how well it respects the user's preferences (*Instantiation Acc*).
3. **Update Plan (Planning Update)** — After every tool call the agent marks the executed step as done and decides whether to replan; *Update Acc* measures preference adherence during replanning.

Each stage reports **Pref Acc**, **Quality Acc**, and **Joint Acc** (both correct simultaneously). A per-difficulty **Pref Score** aggregates the three stages:

```
Pref Score = (First Plan Pref Acc + Update Plan Pref Acc + Instantiation Pref Acc) / 3
```

Preferences are injected in three ways, so an agent can be probed for explicit *and* latent personalization:

- **direct** — explicit user profile is provided.
- **hidden** — only indirect interaction history is provided; preferences must be inferred.
- **mix** — explicit profile and implicit history are provided together.

## 📊 Dataset Statistics

The benchmark pairs a large user population with a single, globally shared offline sandbox so that personalization comes from the *user*, not from per-user environments.

| Scale | Detail |
| --- | --- |
| **Tasks** | 214 total — 68 *easy* + 146 *hard* |
| **Sandbox APIs** | 94 callable APIs across 29 categories (calendar, finance, travel, health, shopping, …) |
| **Real-world records** | ~61.7K candidate records across 6 domains (food, music/media, navigation, shopping, flights, weather) |
| **Preference modes** | 3 (direct / hidden / mix) |
| **Difficulty levels** | Easy, Hard |

Each user directory contains `user_profile.json` (explicit preferences), `hidden_info.json` (indirect preference traces), and `task_easy.json` / `task_hard.json`. The shared environment lives in `SandboxDB/` (API contracts, real-world source records, and a prebuilt vector index for BM25/embedding retrieval).

## 🚀 Quick Start

The project is managed entirely with [`uv`](https://github.com/astral-sh/uv).

### 1. Install environment

```bash
# recommended: reproduce the exact environment from the lockfile
UV_PROJECT_ENVIRONMENT=env uv sync

# or install from the exported package list
pip install -r requirements.txt
```

> `requirements.txt` is a frozen snapshot exported from a Linux + CUDA environment; the authoritative dependency declaration is `pyproject.toml` + `uv.lock`.

### 2. Configure the LLM API

Fill in the API URL, token, and model name in `api.py` before running any evaluation. `api.py` holds both the model-under-test and the LLM-as-judge configuration.

### 3. Evaluate from an API

```bash
python eval/eval_core.py    # run the planning agent over the benchmark
```

User-tunable hyperparameters (difficulty, user/task filters, `PASS_N`, tool-call budget, temperature, concurrency) are exposed at the top of the file. Raw generations are written to `runresult/<model>/<mode>/`, and console logs to `logs/`.

### 4. Judge and summarize

```bash
python eval/judge.py        # LLM-as-judge scoring + HTML report
```

Judging uses `deepseek-v4-flash` by default and emits the three-stage accuracy tables plus an HTML report.

### 5. Evaluate local LLMs

Local offline-inference models are configured under `eval/local/`; see `local_model.py` for the available checkpoints.

## 🗂️ Repository Layout

```
data/         # 214 users: profiles, hidden info, easy/hard tasks
SandboxDB/    # shared sandbox: API contracts, source records, vector index, embedding model
eval/         # eval_core / judge engines + methods (agentic, pag, bank, reason, rag, reminder, local)
Prompts/      # prompt templates (common / eval / judge)
runresult/    # model outputs (git-ignored; placeholder only)
judgeresult/  # judge outputs (git-ignored; placeholder only)
```

## 📚 Citation

If you find this work useful, please ⭐ the repo and cite our paper.

```bibtex
Update When I have the arxiv link
```

## ✅ TODOs

- [ ] Expand the sandbox to more real-world domains

[🔝Back to top](#top)
