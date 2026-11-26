# fedros-omni

A reproducible benchmark for two properties of an LLM agent harness: **task success** and
**hallucination rate**.

```bash
make eval            # offline, free, no API key, no GPU, under a second
make eval-selftest   # checks the scorer itself
```

Results of the recorded run, 3 seeds over 20 tasks:

| Config | Task success | Hallucination rate |
|---|---|---|
| `baseline` - no context, no abstention rule | 0.262 ± 0.109 | 1.000 ± 0.000 |
| `final` - context injected, abstention allowed | 1.000 ± 0.000 | 0.000 ± 0.000 |

**Read [`eval/README.md`](eval/README.md) before quoting any of that.** It gives the exact scoring
rules, the environment, and an honest account of why these particular numbers hit their ceilings: the
runner is a rule-based stub, not a model, so the run proves the harness works and does not measure
agent quality. `make eval-api CONFIRM=yes` is the path to real model numbers.

The same file records which published claims about this project have no run behind them, and where
each one appears.

`tasks/demo-classifier/` is vendored from `maaz-codes/fedros` (lead
[@maaz-codes](https://github.com/maaz-codes)) so that this repository runs standalone. Everything
under `eval/` is this repository's own work. Python 3, standard library only, no dependencies.
