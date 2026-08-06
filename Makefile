.PHONY: help install test demo dry-run calibrate chat ui evals guardrail-ab report clean lint

PY := PYTHONPATH=. python3

help:
	@echo "Ollive evals platform"
	@echo ""
	@echo "  make install       install dependencies"
	@echo "  make test          run the test suite (no API keys needed)"
	@echo "  make dry-run       full eval + report, fully offline (mock arms)"
	@echo "  make calibrate     measure judge quality against gold labels"
	@echo "  make guardrail-ab  quantify what the guardrail layer buys"
	@echo "  make evals         real run: oss vs frontier (needs API keys)"
	@echo "  make chat          interactive CLI (VARIANT=mock|oss|frontier)"
	@echo "  make ui            Streamlit chat UI"
	@echo "  make clean         remove runs/, reports/, caches"

install:
	pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m pyflakes src tests 2>/dev/null || echo "(pyflakes not installed — skipping)"

# ---- offline: proves the whole pipeline works with zero credentials ---------
dry-run:
	$(PY) -m src.evals.cli run \
		--arms mock-strong,mock-weak \
		--mock-judge \
		--report reports/dry-run.html
	@echo "\n-> reports/dry-run.html"

calibrate:
	$(PY) -m src.evals.cli calibrate --heuristic-judge
	@echo "\n(add --mock-judge, or drop the flag entirely to use the real Claude judge)"

guardrail-ab:
	$(PY) scripts/guardrail_ab.py

# ---- real run: needs ANTHROPIC_API_KEY and TOGETHER_API_KEY (or HF_TOKEN) ---
evals:
	$(PY) -m src.evals.cli run \
		--arms oss,frontier \
		--judge-samples 1 \
		--report reports/evaluation.html
	@echo "\n-> reports/evaluation.html"

evals-full:
	$(PY) -m src.evals.cli run \
		--arms oss,frontier \
		--judge-samples 3 \
		--concurrency 4 \
		--report reports/evaluation.html

judge-quality:
	$(PY) -m src.evals.cli calibrate --json > reports/judge_quality.json
	@echo "-> reports/judge_quality.json"

dataset:
	$(PY) -m src.evals.cli dataset

VARIANT ?= mock
chat:
	$(PY) -m src.wellness.cli chat --variant $(VARIANT) --trace

ui:
	streamlit run src/wellness/ui/streamlit_app.py

clean:
	rm -rf runs reports .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
