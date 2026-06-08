# example_needle — developer tasks
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test cover clean build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (core deps only)
	python -m pip install -e .

install-dev: ## Install with dev + test tooling
	python -m pip install -e ".[dev]"

lint: ## Run ruff + black in check mode
	ruff check src tests examples scripts
	black --check src tests examples scripts

format: ## Auto-format with ruff --fix + black
	ruff check --fix src tests examples scripts
	black src tests examples scripts

typecheck: ## Run mypy over the source tree
	mypy src

test: ## Run the test suite
	pytest

cover: ## Run tests with coverage
	pytest --cov=needle_core --cov=needle --cov-report=term-missing

build: ## Build sdist + wheel
	python -m build

clean: ## Remove build / cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
