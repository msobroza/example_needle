# rag_load_test — developer tasks (run from this directory)
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test ingest serve loadtest compare ovms-export infra-up infra-down

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
	ruff check src tests locustfile.py
	black --check src tests locustfile.py

format: ## Auto-format with ruff --fix + black
	ruff check --fix src tests locustfile.py
	black src tests locustfile.py

typecheck: ## Run mypy over the source tree
	mypy src

test: ## Run the test suite (no model weights, no network)
	pytest

ingest: ## Index the synthetic corpus into Elasticsearch (RAG_ES_URL)
	rag-ingest --synthetic 200

serve: ## Run the FastAPI service with the current RAG_* environment
	rag-serve

loadtest: ## Headless locust run: make loadtest RUN_NAME=monolith HOST=http://host:8888
	mkdir -p results
	locust --headless --host $(HOST) -u $(USERS) -r $(SPAWN_RATE) -t $(RUN_TIME) \
		--csv results/$(RUN_NAME) --html results/$(RUN_NAME).html --only-summary

compare: ## Compare all runs found under results/ into results/comparison.md
	rag-compare $(sort $(patsubst %_stats.csv,%,$(wildcard results/*_stats.csv))) --out results/comparison.md

ovms-export: ## Export reranker + embedder to ovms/models (needs HF access)
	./ovms/export_models.sh

infra-up: ## Start Elasticsearch (9200) + OVMS reranker (8001) + embedder (8002)
	docker compose --env-file ovms/versions.env up -d

infra-down: ## Stop the local containers
	docker compose --env-file ovms/versions.env down
