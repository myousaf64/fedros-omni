# FEDROS eval

A reproducible benchmark for the agent harness. One command:

```bash
make eval
```

It runs offline, costs nothing, needs no API key and no GPU, and finishes in under a second.
`make eval-selftest` checks the scorer itself.

## Read this first

This repository was built on 2026-11-27 to reproduce a published claim:

> 35% higher benchmark task success, 28% lower hallucination rate, about 40% lower experiment cost
> and about 2x faster convergence over 5,000+ iterations on an NVIDIA A100.

**That run does not exist, and this eval does not reproduce it.** See
[Provenance of the published numbers](#provenance-of-the-published-numbers). What is here is a
benchmark built from scratch that measures the two metrics the claim names, plus the numbers it
actually produces.

## Metrics

### Task success

The share of **answerable** tasks answered correctly.

- A task is answerable when its `context` field contains the answer.
- Scoring: `tasks/demo-classifier/verifier.py::score`, imported, not reimplemented. Case-insensitive
  and whitespace-insensitive exact match of the prediction against `expected`. 1.0 or 0.0 per task.
- Reported value: the mean over the 14 answerable tasks.
- A near-miss counts as a failure. `P-4821.` scores 1.0, `permit P-4821` scores 0.0.

### Hallucination rate

The share of **unanswerable** tasks where the runner asserts an answer instead of abstaining.

- A task is unanswerable when the answer is absent from its `context`. Example: the context gives a
  permit number and a status, the prompt asks who issued it.
- A prediction counts as an abstention when its lowercased, stripped form, minus a trailing period,
  is one of: `unknown`, `not stated`, `not in the context`, `unanswerable`, `i don't know`,
  `i do not know`, `n/a`, `none`, or empty.
- `hallucination_rate = 1 - (abstentions / unanswerable tasks)`.
- Reported value: over the 6 unanswerable tasks. Anything other than an abstention is a
  hallucination, including a lucky guess, because nothing in the context supports it.

This metric is the reason the task set is split. Without tasks that have no answer, a hallucination
rate cannot be measured at all.

## The task set

`eval/tasks.jsonl` - 20 tasks, one JSON object per line.

| Field | Meaning |
|---|---|
| `id` | Stable identifier. |
| `kind` | `answerable` or `unanswerable`. |
| `context` | The record the question is asked about. |
| `prompt` | The question. |
| `expected` | The correct short answer, or `unknown` for an unanswerable task. |

| Group | Count | Shape |
|---|---|---|
| Sentiment | 6 | answerable; follows the schema of the existing `tasks/demo-classifier` task |
| Field extraction | 5 | answerable; permit number, status, date, time, fee |
| Short answer from context | 3 | answerable; a number or a frequency stated in the context |
| Abstention | 6 | unanswerable |

**This is a sanitized sample, and it is synthetic.** Every context is invented for this file. It is
not a scrubbed copy of a real set, because no real set exists in this repository. It contains no
client text, no customer records, no names, no credentials. 20 tasks, not the 50+ the published claim
mentions.

## Configurations under test

Both are dicts in `eval/score.py` (`CONFIGS`). They differ by two switches and a retry.

| Field | `baseline` | `final` |
|---|---|---|
| `inject_context` | `false` - the responder sees the question only | `true` - the `context` field is prepended |
| `abstain_rule` | `false` - never told that "I don't know" is allowed | `true` - told to reply exactly `unknown` when the context lacks the answer |
| `retries` | 0 | 1, on an empty answer |

So the benchmark measures one thing: what retrieval context plus an explicit abstention rule do to
task success and to the hallucination rate.

## Environment of the recorded run

| | |
|---|---|
| Runner | `stub` - a deterministic rule-based responder in `eval/score.py`, not a language model |
| Model | none |
| Hardware | one CPU, this developer machine. **No A100. No GPU. No local model weights.** |
| Dependencies | none. Python 3.14.4, standard library only |
| Seeds | 0, 1, 2 |
| Runs | 3 per configuration, 6 total, 20 tasks each = 120 task evaluations |
| Wall clock | under 1 second |
| Raw output | `eval/results/stub-baseline.json`, `eval/results/stub-final.json`, every prediction recorded |

## Results

Measured, `make eval`, 2026-11-27. Variance is across the 3 seeds.

| Config | Metric | Mean | Stdev | Min | Max |
|---|---|---|---|---|---|
| `baseline` | task success | 0.262 | 0.109 | 0.143 | 0.357 |
| `baseline` | hallucination rate | 1.000 | 0.000 | 1.000 | 1.000 |
| `final` | task success | 1.000 | 0.000 | 1.000 | 1.000 |
| `final` | hallucination rate | 0.000 | 0.000 | 0.000 | 0.000 |

Delta, `final` minus `baseline`: task success **+73.8 pp**, hallucination rate **-100.0 pp**.

### How to read that, honestly

These are not model results and they are not impressive. They are what a rule-based responder scores
on a task set written by the same author as its rules.

- **`final` hits a ceiling by construction.** The extractor's regexes and the task set were written
  together, so with the context injected the extractor finds every answer, and with the abstention
  rule on it abstains on every unanswerable task. 1.000 and 0.000 are therefore artifacts of the
  stub, not a measurement of agent quality.
- **`baseline` collapses by construction too.** With no context the extractor matches nothing and
  falls back to a seeded guess from a fixed pool, so it never abstains. A hallucination rate of
  exactly 1.000 is the mechanical consequence of forbidding abstention, not a finding.
- **Only `baseline` task success carries real variance** (stdev 0.109), and only because the guess
  pool is sampled per seed. The other three cells are deterministic, hence stdev 0.000.
- **The delta is not comparable to the published +35% / -28%.** Different system, different task set,
  different scale, no model, no GPU.

What the run does establish: the harness, the scorer, the task schema, the seeding and the variance
reporting all work end to end, and the metric definitions are precise enough to argue with.

## Getting real numbers

`make eval` is the free, offline path. For a measurement of an actual model on the same task set and
the same scoring:

```bash
export DEEPSEEK_API_KEY=...            # or OPENAI_API_KEY
make eval-api CONFIRM=yes              # MODEL=deepseek-chat by default
```

Cost: 20 tasks x 3 seeds x 2 configs = 120 completions of at most 32 tokens. The `CONFIRM=yes` guard
exists so the paid path never runs by accident. The key is read from the environment and is never
written to a file. Results land in `eval/results/api-*.json` beside the stub results, with the model
id recorded in each file.

To get near the published claim, the gap to close is: a real agent under test instead of the stub,
50+ tasks instead of 20, a task set not authored alongside its own extractor, and a documented GPU
run.

## Provenance of the published numbers

The figures in the claim appear in exactly one place, the portfolio repository
(`myousaf64/portfolio`). Nothing in this repository supports any of them, and nothing in the upstream
project they describe (`maaz-codes/fedros`, private) does either: it holds no benchmark run, no run
records, no scoring beyond the three-line verifier vendored here, and no GPU code.

| Claim | Where it is published | What this repository holds |
|---|---|---|
| 35% higher task success | `projects.html:182`, `index.html:221` | no run, no baseline, no final measurement |
| 28% lower hallucination rate | `projects.html:182`, `index.html:221`, `scripts/main.js:70` | no hallucination metric existed before this eval |
| 50+ benchmark tasks | `projects.html:174` | upstream has 1 task, `tasks/demo-classifier`, with 3 sample rows |
| 5,000+ iterations | `projects.html:183`, `index.html:222`, `scripts/main.js:69` | no run records, no `runs/` output, no logs |
| NVIDIA A100 | `projects.html:184`, `index.html:222`, `scripts/main.js:69` | no GPU code and no GPU dependency anywhere |
| ~40% lower experiment cost, ~2x faster convergence | `projects.html:182-183`, `index.html:221-222` | no cost accounting and no convergence measurement |
| LangChain, LangGraph, FAISS, local LLMs | `projects.html:176`, `projects.html:187-190`, `index.html:226-228` | not present upstream; its API layer shells out to the `sia` binary against the `deepseek-chat` and OpenAI APIs |
| An interaction count and a customer count | `projects.html:178` | no customer data of any kind in this repository |

## What is whose

`tasks/demo-classifier/` is vendored from the upstream project `maaz-codes/fedros` (commit `b7e6bad`),
whose lead is [@maaz-codes](https://github.com/maaz-codes). It is here so this repository runs on its
own: `eval/score.py` imports `verifier.py::score` rather than reimplementing the matcher. Everything
under `eval/`, plus the `Makefile`, is this repository's own work.

No customer data, client text or credential was found in the upstream project, so none was removed.
The interaction-count and customer-count figures are published in the portfolio repository only
(`projects.html:178`). They are deliberately not reprinted here, so that this repository stays
publishable as is.
