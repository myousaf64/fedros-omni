.PHONY: eval eval-selftest eval-api

# Offline, free, deterministic. This is the one command to rerun the benchmark.
eval:
	python3 eval/score.py --runner stub --seeds 0,1,2

eval-selftest:
	python3 eval/score.py --selftest

# Paid path. Needs MODEL and the matching API key in the environment.
# Guarded so it never runs by accident: make eval-api CONFIRM=yes
eval-api:
	@test "$(CONFIRM)" = "yes" || { echo "this path spends API credit. rerun: make eval-api CONFIRM=yes"; exit 1; }
	MODEL=$(or $(MODEL),deepseek-chat) python3 eval/score.py --runner api --seeds 0,1,2
