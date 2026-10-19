.PHONY: install lint test experiments figures receipts receipts-fast verify clean

# Every target runs through one interpreter. Calling a bare `pytest` picks whichever
# console script happens to be first on PATH, which on a fresh machine can belong to a
# different Python than the one holding the editable install. That failure looks like a
# missing package and is not one: `make test` failed in a clean clone of this repository
# while `python -m pytest` in the same directory passed.
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest --cov=cost_thresholds --cov-report=term-missing --cov-fail-under=100

experiments:
	$(PYTHON) experiments/e1_invariance.py
	$(PYTHON) experiments/e2_cost_ratio.py
	$(PYTHON) experiments/e3_repair.py
	$(PYTHON) experiments/e4_cheap_screen.py

figures:
	$(PYTHON) tools/render_diagram.py
	$(PYTHON) tools/capture_cli.py

# Re-measures everything, then refuses to pass if a document quotes a figure the code no
# longer produces. Use receipts-fast locally to skip the experiment re-run.
receipts:
	$(PYTHON) tools/collect_metrics.py
	$(PYTHON) tools/check_numbers.py

receipts-fast:
	$(PYTHON) tools/collect_metrics.py --reuse
	$(PYTHON) tools/check_numbers.py

verify: lint test receipts

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.json junit.xml .coverage
