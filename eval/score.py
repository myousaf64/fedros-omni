#!/usr/bin/env python3
"""Reproducible benchmark for the FEDROS agent harness.

Measures two metrics over the task set in eval/tasks.jsonl:

  task success      mean verifier score over the answerable tasks
  hallucination     share of the unanswerable tasks where the runner asserts
                    an answer instead of abstaining

It compares two harness configurations, `baseline` and `final`, over several
seeds, and prints the mean and the variance across runs. See eval/README.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = Path(__file__).resolve().parent / "tasks.jsonl"
RESULTS = Path(__file__).resolve().parent / "results"

# Reuse the verifier that already ships with the demo task instead of writing
# a second matcher.
_spec = importlib.util.spec_from_file_location(
    "demo_verifier", ROOT / "tasks" / "demo-classifier" / "verifier.py"
)
_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_verifier)
score_answer = _verifier.score

ABSTAIN = {"unknown", "not stated", "not in the context", "unanswerable",
           "i don't know", "i do not know", "n/a", "none", ""}

CONFIGS = {
    # The harness as it starts out: the model sees the question only, and is
    # never told that "I don't know" is an acceptable answer.
    "baseline": {"inject_context": False, "abstain_rule": False, "retries": 0},
    # The harness after the two changes under test.
    "final": {"inject_context": True, "abstain_rule": True, "retries": 1},
}

SYSTEM_BASE = "You answer questions about government service records. Answer with the shortest possible span, no sentence."
SYSTEM_ABSTAIN = ' If the answer is not present in the context, reply exactly "unknown".'

POSITIVE = ("excellent", "helpful", "great", "polite", "on time", "two minutes")
NEGATIVE = ("terrible", "not recommend", "dirty", "slow", "rejected", "nobody answered")

# prompt pattern -> how to pull the answer out of the context
RULES = (
    (r"sentiment", lambda c: _sentiment(c)),
    (r"permit number", lambda c: _find(r"permit number (P-\d+)", c)),
    (r"permit status", lambda c: _find(r"status (active|expired|pending)", c)),
    (r"ticket closed", lambda c: _find(r"closed on (\d{4}-\d{2}-\d{2})", c)),
    (r"closing time", lambda c: _find(r"closes at (\d{2}:\d{2})", c)),
    (r"renewal fee", lambda c: _find(r"costs (\d+ AED)", c)),
    (r"days.*appeal", lambda c: _find(r"within (\d+) days", c)),
    (r"how often", lambda c: "quarterly" if "quarter" in c.lower() else None),
    (r"age in years", lambda c: _find(r"older than (\d+) years", c)),
)

# What a responder with no context and no permission to abstain says instead.
GUESSES = {
    "sentiment": ["positive", "negative"],
    "permit number": ["P-0001", "P-1234", "P-5000"],
    "permit status": ["active", "pending"],
    "ticket closed": ["2026-01-01", "2026-06-30"],
    "closing time": ["17:00", "16:30"],
    "renewal fee": ["100 AED", "500 AED"],
    "days.*appeal": ["14", "60"],
    "how often": ["monthly", "yearly"],
    "age in years": ["5", "10"],
    "officer": ["Officer on duty", "The issuing officer"],
    "appeal fee": ["50 AED", "no fee"],
    "saturday": ["08:00 to 12:00", "closed"],
    "expedited": ["400 AED", "double the standard fee"],
    "department": ["the service desk", "operations"],
    "stars": ["5", "4"],
}


def _find(pattern: str, context: str) -> str | None:
    m = re.search(pattern, context, re.I)
    return m.group(1) if m else None


def _sentiment(context: str) -> str | None:
    low = context.lower()
    if any(w in low for w in POSITIVE):
        return "positive"
    if any(w in low for w in NEGATIVE):
        return "negative"
    return None


def extract(prompt: str, context: str) -> str | None:
    """Answer the prompt from the context, or None when the context lacks it."""
    for pattern, rule in RULES:
        if re.search(pattern, prompt, re.I):
            return rule(context)
    return None


def _guess(prompt: str, rng: random.Random) -> str:
    for key, pool in GUESSES.items():
        if re.search(key, prompt, re.I):
            return rng.choice(pool)
    return rng.choice(["yes", "no"])


def run_stub(task: dict, cfg: dict, rng: random.Random) -> str:
    """Offline rule-based responder. Measures the harness, not a model."""
    context = task["context"] if cfg["inject_context"] else ""
    answer = extract(task["prompt"], context)
    for _ in range(cfg["retries"]):
        if answer:
            break
        answer = extract(task["prompt"], context)
    if answer is None:
        return "unknown" if cfg["abstain_rule"] else _guess(task["prompt"], rng)
    return answer


def run_api(task: dict, cfg: dict, rng: random.Random) -> str:
    """Paid path. Needs DEEPSEEK_API_KEY (or OPENAI_API_KEY) in the environment."""
    model = os.environ.get("MODEL", "deepseek-chat")
    if "deepseek" in model:
        url, key = "https://api.deepseek.com/chat/completions", os.environ.get("DEEPSEEK_API_KEY")
    else:
        url, key = "https://api.openai.com/v1/chat/completions", os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("no API key in the environment for model %r" % model)
    system = SYSTEM_BASE + (SYSTEM_ABSTAIN if cfg["abstain_rule"] else "")
    user = task["prompt"]
    if cfg["inject_context"]:
        user = "Context: %s\n\n%s" % (task["context"], user)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0, "seed": rng.randint(0, 2**31 - 1), "max_tokens": 32,
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    for attempt in range(cfg["retries"] + 1):
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.load(resp)["choices"][0]["message"]["content"].strip()
        if out:
            return out
        del attempt
    return ""


RUNNERS = {"stub": run_stub, "api": run_api}


def is_abstain(prediction: str) -> bool:
    return prediction.strip().lower().rstrip(".") in ABSTAIN


def run_once(tasks: list[dict], runner, cfg: dict, seed: int) -> dict:
    rng = random.Random(seed)
    answerable, abstained, records = [], [], []
    for task in tasks:
        prediction = runner(task, cfg, rng)
        if task["kind"] == "answerable":
            hit = score_answer(prediction, task["expected"])
            answerable.append(hit)
        else:
            hit = float(is_abstain(prediction))
            abstained.append(hit)
        records.append({"id": task["id"], "kind": task["kind"],
                        "prediction": prediction, "expected": task["expected"],
                        "scored": hit})
    return {
        "seed": seed,
        "task_success": statistics.fmean(answerable),
        "hallucination_rate": 1.0 - statistics.fmean(abstained),
        "records": records,
    }


def summarize(runs: list[dict], metric: str) -> dict:
    values = [r[metric] for r in runs]
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def selftest() -> None:
    cfg_final, cfg_base = CONFIGS["final"], CONFIGS["baseline"]
    rng = random.Random(0)
    answerable = {"id": "t", "kind": "answerable",
                  "context": "Permit record: permit number P-4821, status active.",
                  "prompt": "Give the permit number only.", "expected": "P-4821"}
    unanswerable = {"id": "u", "kind": "unanswerable",
                    "context": "Permit record: permit number P-4821, status active.",
                    "prompt": "Give the name of the officer who issued the permit.",
                    "expected": "unknown"}
    assert run_stub(answerable, cfg_final, rng) == "P-4821"
    assert run_stub(unanswerable, cfg_final, rng) == "unknown"
    assert not is_abstain(run_stub(unanswerable, cfg_base, rng)), "baseline must assert, not abstain"
    assert is_abstain("Unknown.") and is_abstain("") and not is_abstain("P-4821")
    assert score_answer("P-4821", "p-4821 ") == 1.0 and score_answer("P-0001", "P-4821") == 0.0
    good = run_once([answerable, unanswerable], run_stub, cfg_final, 0)
    assert good["task_success"] == 1.0 and good["hallucination_rate"] == 0.0
    bad = run_once([answerable, unanswerable], run_stub, cfg_base, 0)
    assert bad["task_success"] == 0.0 and bad["hallucination_rate"] == 1.0
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="stub", choices=sorted(RUNNERS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--configs", default="baseline,final")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    runner = RUNNERS[args.runner]
    RESULTS.mkdir(exist_ok=True)

    summary = {}
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        runs = [run_once(tasks, runner, cfg, seed) for seed in seeds]
        summary[name] = {metric: summarize(runs, metric)
                         for metric in ("task_success", "hallucination_rate")}
        out = RESULTS / ("%s-%s.json" % (args.runner, name))
        out.write_text(json.dumps({
            "runner": args.runner,
            "config_name": name,
            "config": cfg,
            "model": os.environ.get("MODEL", "rule-based stub") if args.runner == "api" else "rule-based stub",
            "task_file": TASKS.name,
            "n_tasks": len(tasks),
            "seeds": seeds,
            "summary": summary[name],
            "runs": runs,
        }, indent=2) + "\n")
        print("wrote %s" % out.relative_to(ROOT))

    print("\nrunner: %s   tasks: %d   seeds: %s   runs per config: %d"
          % (args.runner, len(tasks), seeds, len(seeds)))
    header = "%-10s %-20s %8s %8s %8s %8s" % ("config", "metric", "mean", "stdev", "min", "max")
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in summary.items():
        for metric, stats in metrics.items():
            print("%-10s %-20s %8.3f %8.3f %8.3f %8.3f"
                  % (name, metric, stats["mean"], stats["stdev"], stats["min"], stats["max"]))
    if {"baseline", "final"} <= summary.keys():
        print("\ndelta final - baseline (percentage points):")
        for metric in ("task_success", "hallucination_rate"):
            delta = summary["final"][metric]["mean"] - summary["baseline"][metric]["mean"]
            print("  %-20s %+.1f pp" % (metric, delta * 100))


if __name__ == "__main__":
    main()
