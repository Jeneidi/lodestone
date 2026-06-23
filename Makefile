# =============================================================================
# Lodestone — project automation
# =============================================================================
# All targets use .venv/bin/ prefixes so they work without activating the venv.
# Run `make install` first, then any other target.
#
# Usage:
#   make install      Create venv and install all dependencies
#   make data         Download and preprocess the evaluation dataset
#   make lint         Run ruff linter
#   make format       Run ruff formatter (in-place)
#   make typecheck    Run mypy over the src package
#   make test         Run pytest (quiet mode)
#   make eval         Run the evaluation harness
#   make ablate       Run the ablation sweep
#   make serve        Start the FastAPI development server
#   make dashboard    Start the Streamlit dashboard
# =============================================================================

VENV        := .venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
MYPY        := $(VENV)/bin/mypy
UVICORN     := $(VENV)/bin/uvicorn
STREAMLIT   := $(VENV)/bin/streamlit

.PHONY: help install data lint format typecheck test eval ablate serve dashboard clean

# Default target
help:
	@echo ""
	@echo "  Lodestone — available make targets"
	@echo "  -----------------------------------"
	@echo "  install    Create .venv and install all dependencies (run first)"
	@echo "  data       Download + preprocess the SQuAD-derived evaluation dataset"
	@echo "  lint       Run ruff linter (no changes)"
	@echo "  format     Run ruff formatter (modifies files in-place)"
	@echo "  typecheck  Run mypy static type checker over src/lodestone"
	@echo "  test       Run pytest -q (all tests, offline-friendly)"
	@echo "  eval       Run the evaluation harness  (python -m evals.harness)"
	@echo "  ablate     Run ablation sweep          (python -m evals.ablation)"
	@echo "  serve      Start FastAPI dev server on http://127.0.0.1:8000"
	@echo "  dashboard  Start Streamlit dashboard   on http://localhost:8501"
	@echo "  clean      Remove .venv, __pycache__, and build artifacts"
	@echo ""

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

install:
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "  Installation complete.  Activate with: source $(VENV)/bin/activate"
	@echo "  Or run targets directly — all Makefile targets use $(VENV)/bin/ prefixes."
	@echo ""

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

data:
	$(PYTHON) scripts/build_dataset.py

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	$(RUFF) check src/ evals/ tests/

format:
	$(RUFF) format src/ evals/ tests/ scripts/
	$(RUFF) check --fix src/ evals/ tests/

typecheck:
	$(MYPY) src/lodestone

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	$(PYTEST) -q

test-cov:
	$(PYTEST) -q --cov=lodestone --cov-report=term-missing --cov-report=html

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval:
	$(PYTHON) -m evals.harness

ablate:
	$(PYTHON) -m evals.ablation

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

serve:
	$(UVICORN) lodestone.api.server:app --reload --host 127.0.0.1 --port 8000

dashboard:
	$(STREAMLIT) run dashboard/app.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist"          -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
