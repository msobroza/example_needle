# rag_load_test — developer tasks (run from this directory)
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test ingest serve loadtest compare ovms-export ovms-up ovms-down

PYTHON ?= python
RUN_NAME ?= monolith
HOST ?= http://localhost:8888
USERS ?= 20
SPAWN_RATE ?= 5
RUN_TIME ?= 2m

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the service (runtime deps only)
	$(PYTHON) -m pip install -e .

install-dev: ## Install with locust + test tooling
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff + black in check mode
	ruff check src tests loadtest scripts
	black --check src tests loadtest scripts

format: ## Auto-format with ruff --fix + black
	ruff check --fix src tests loadtest scripts
	black src tests loadtest scripts

typecheck: ## Run mypy over the source tree
	mypy src

test: ## Run the test suite (no model weights, no network)
	pytest

ingest: ## Build the local Chroma index from the synthetic corpus
	rag-ingest --synthetic 200

serve: ## Run the FastAPI service with the current RAG_* environment
	rag-serve

loadtest: ## Headless locust run: make loadtest RUN_NAME=monolith HOST=http://...:8888
	./scripts/run_loadtest.sh $(RUN_NAME) $(HOST) $(USERS) $(SPAWN_RATE) $(RUN_TIME)

compare: ## Compare all runs found under results/ into results/comparison.md
	$(PYTHON) scripts/compare_runs.py $(sort $(patsubst %_stats.csv,%,$(wildcard results/*_stats.csv))) --out results/comparison.md

ovms-export: ## Export reranker + embedder to ovms/models (needs HF access)
	./ovms/export_models.sh

ovms-up: ## Start local OVMS reranker (8001) + embedder (8002)
	docker compose -f ovms/docker-compose.yml --env-file ovms/versions.env up -d

ovms-down: ## Stop local OVMS containers
	docker compose -f ovms/docker-compose.yml --env-file ovms/versions.env down
