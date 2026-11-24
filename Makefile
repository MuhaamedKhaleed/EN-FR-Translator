.PHONY: install test lint check
install:
	uv sync --locked --extra dev --extra ml
test:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 uv run pytest --cov=enfr_translator --cov-report=term-missing
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src
check: lint test
	uv run pre-commit run --all-files
