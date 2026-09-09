# Deploying the three topologies on Domino 6.2

The same FastAPI + LangGraph service is published as one, two or three Domino
Apps. Only environment variables differ between topologies, so the numbers
Locust produces are comparable.

Every Domino App must listen on `0.0.0.0:8888`; each entry-point script exports
`RAG_HOST=0.0.0.0 RAG_PORT=8888` (or passes `--rest_port 8888` to `ovms`) before
`exec`-ing the server. Domino selects the entry point when you publish the App.

## Topologies, apps, entry points

| Topology | App | Entry point | Environment (`RAG_*`) | Environment image |
| --- | --- | --- | --- | --- |
| `monolith` | one app: workflow + embedder + Chroma + reranker | `domino/app_monolith.sh` | `OPENAI_API_KEY`; optional `RAG_TOPOLOGY` label (default `monolith`) | `Dockerfile.rag` |
| `split-reranker` | **A** OVMS reranker | `domino/app_ovms_reranker.sh` | `RAG_OVMS_MODELS_DIR` (default `/mnt/data/ovms-models`) | `Dockerfile.ovms` |
| | **B** workflow + embedder + Chroma | `domino/app_workflow.sh` | `RAG_OVMS_RERANK_URL=<URL of A>`; `RAG_OVMS_AUTH=domino` (default); `OPENAI_API_KEY` | `Dockerfile.rag` |
| `split-all` | **A** OVMS reranker | `domino/app_ovms_reranker.sh` | as above | `Dockerfile.ovms` |
| | **C** OVMS embedder | `domino/app_ovms_embedder.sh` | `RAG_OVMS_MODELS_DIR` | `Dockerfile.ovms` |
| | **B** workflow + Chroma | `domino/app_workflow.sh` | `RAG_OVMS_RERANK_URL=<URL of A>`; `RAG_EMBEDDER_BACKEND=ovms`; `RAG_OVMS_EMBEDDINGS_URL=<URL of C>`; `OPENAI_API_KEY` | `Dockerfile.rag` |

`app_workflow.sh` derives `RAG_TOPOLOGY` (`split-reranker`, or `split-all` when
`RAG_EMBEDDER_BACKEND=ovms`) unless you set it. The label is echoed in every
response (`deployment` field, `X-RAG-Topology` header) and ends up in the
Locust run name, so use distinct labels when you test several hardware tiers.

## 1. Compute environments

Two environments, built from `domino/environment/`:

| File | Contents | Used by |
| --- | --- | --- |
| `Dockerfile.rag` | `pip install rag_load_test[loadtest,domino]` on the Domino Standard Environment, both HF models pre-downloaded, `HF_HUB_OFFLINE=1` | the RAG apps, Workspaces that run Locust |
| `Dockerfile.ovms` | the pinned `ovms` binary tarball unpacked under `/opt/ovms` (`PATH`, `LD_LIBRARY_PATH` set) | the OVMS apps |

Build from the repository root (the context must contain `rag_load_test/`):

```bash
docker build -f rag_load_test/domino/environment/Dockerfile.rag -t <registry>/rag-load-test-env .
docker build -f rag_load_test/domino/environment/Dockerfile.ovms \
  --build-arg "OVMS_BINARY_URL=$(grep '^OVMS_BINARY_URL=' rag_load_test/ovms/versions.env | cut -d= -f2-)" \
  -t <registry>/rag-load-test-ovms-env .
```

Push the images and create two Domino Environments that use them as custom
base images (Environments → Create Environment → Custom Image). Alternatively
pick `quay.io/domino/compute-environment-images:ubuntu22-py3.10-r4.4-domino6.2-standard`
as the base and paste the instructions after `FROM` into the environment's
Dockerfile Instructions field.

## 2. Exported models in a Domino Dataset

The OVMS apps read `config_reranker.json` / `config_embedder.json` from
`RAG_OVMS_MODELS_DIR`, default `/mnt/data/ovms-models`, and exit with a clear
message when the file is missing.

1. Create a Dataset named `ovms-models` in the project (Data → Datasets).
   Git-based projects mount it at `/mnt/data/ovms-models`; projects using the
   Domino File System mount it at `/domino/datasets/local/ovms-models`, so set
   `RAG_OVMS_MODELS_DIR` accordingly.
2. Fill it, either by exporting directly into it from a Workspace that uses the
   `Dockerfile.rag` environment:

   ```bash
   cd rag_load_test && ./ovms/export_models.sh /mnt/data/ovms-models
   ```

   or by running `make ovms-export` on a laptop and uploading the contents of
   `ovms/models/` (configs plus the two model directories) into the Dataset.
3. Attach the Dataset to the OVMS apps (Apps inherit the project's Dataset
   mounts).

## 3. Publish an App

For every app in the table above (Apps view → Publish / App Setup):

1. Entry point: the script path, e.g. `rag_load_test/domino/app_workflow.sh`
   (the scripts `cd` to `rag_load_test/` themselves).
2. Environment: `Dockerfile.rag` image for RAG apps, `Dockerfile.ovms` image
   for OVMS apps.
3. Hardware tier: a CPU tier with at least 4 cores (the int8 OVMS models and
   the in-process models are CPU-bound); 8 cores for the monolith is a fair
   baseline. Set `RAG_MODEL_THREADS` to the core count for the RAG apps.
4. Environment variables: Project → Settings → Environment variables (Apps
   inherit them); per-app values such as `RAG_OVMS_RERANK_URL` go there as well,
   with one project per app if two apps of the same project need different
   values.
5. Permissions: the OVMS apps must be reachable by the user who owns the
   workflow app's run, and the workflow app by whoever runs Locust
   ("Anyone with an account" is the simplest choice for a benchmark).

Publish the OVMS apps first, copy their URLs from the Apps view, then publish
the workflow app with those URLs.

### App URLs and the reverse proxy (`DOMINO_RUN_HOST_PATH`)

Domino serves an App under a path prefix and exposes that prefix to the process
as `DOMINO_RUN_HOST_PATH`. The proxy strips the prefix before forwarding, so the
routes stay at `/`; the FastAPI app only uses `DOMINO_RUN_HOST_PATH` as
`root_path` so `/docs` and OpenAPI links resolve. Consequences:

* Calling an app from outside means prefixing every route with the App URL:
  `https://<domino-host><DOMINO_RUN_HOST_PATH>/query`.
* `RAG_OVMS_RERANK_URL` / `RAG_OVMS_EMBEDDINGS_URL` are the OVMS App URLs
  **without a trailing slash**; the adapters append `/v3/rerank` and
  `/v3/embeddings`.

### App-to-app authentication

Domino apps only accept authenticated requests. Inside a Domino run a short
lived bearer token is available from `http://localhost:8899/access-token`.
`app_workflow.sh` sets `RAG_OVMS_AUTH=domino`, which makes the workflow app fetch
that token (endpoint configurable via `RAG_DOMINO_ACCESS_TOKEN_URL`), cache it
for 4 minutes and refresh it on `401`, and send it as `Authorization: Bearer`
on every OVMS call. To use a fixed token instead, set `RAG_OVMS_AUTH=static` and
`RAG_OVMS_BEARER_TOKEN`.

## 4. Deploy as an Agent (tracing)

Agent deployments use the same hosting as Apps (same entry point script, same
`0.0.0.0:8888` rule) and add tracing through the `@add_tracing` decorator from
`dominodatalab[agents]` (the package's `[domino]` extra, already installed by
`Dockerfile.rag`).

1. Set `RAG_DOMINO_TRACING=true`: the `/query` handler is wrapped with
   `add_tracing(name="rag_query", autolog_frameworks=["langchain"])`, so the
   LangGraph run and the OpenAI call show up as traces. When the package is not
   importable the wrapper logs a warning and stays a no-op.
2. Experiment Manager → **Deploy Agent** → choose the entry point
   (`domino/app_monolith.sh` or `domino/app_workflow.sh`), environment and
   hardware tier exactly as for an App.
3. Load test it the same way; tracing adds a small per-request overhead, so keep
   the flag identical across the runs you compare.

## 5. Running Locust against the apps

### From a Domino Workspace (same project, `Dockerfile.rag` environment)

```bash
cd rag_load_test
export RAG_LOADTEST_BEARER_TOKEN=$(curl -s http://localhost:8899/access-token)
make loadtest RUN_NAME=monolith        HOST=https://<domino-host><path of the monolith app>
make loadtest RUN_NAME=split-reranker  HOST=https://<domino-host><path of workflow app B>
make loadtest RUN_NAME=split-all       HOST=https://<domino-host><path of workflow app B, split-all project>
make compare                           # results/comparison.md
```

The token expires after about 5 minutes and Locust reads it once per user at
start, so for runs longer than that (`RUN_TIME=10m`, or `RAG_LOADTEST_SHAPE=step`)
use an API key instead.

### From a laptop

```bash
export DOMINO_API_KEY=<Account settings → API Key>   # sent as X-Domino-Api-Key
make loadtest RUN_NAME=monolith HOST=https://<domino-host><path of the monolith app>
```

`HOST` must not end with a slash: Locust concatenates it with `/query` and
`/retrieve`. Per-stage latencies come back in `timings_ms` and are recorded as
extra `STAGE` rows (`/query:embed`, `/query:rerank`, ...) in
`results/<run>_stats.csv`.

## 6. Checklist per run

1. `GET <app URL>/readyz` returns `200` and lists `vector_store`, `embedder`
   (and `reranker`) as `ok` (`503` shows which check failed).
2. Same corpus in every workflow app: run `rag-ingest --synthetic 200` once per
   app (or share `RAG_CHROMA_PATH` through a Dataset) with the embedder backend
   that app will serve with.
3. Same `USERS`, `SPAWN_RATE`, `RUN_TIME` and hardware tier across topologies.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `401`/`403` from Locust | missing or expired `RAG_LOADTEST_BEARER_TOKEN` / `DOMINO_API_KEY`, or the app is not shared with that user |
| `404` on `/query` | `HOST` lacks the App path prefix, or has a trailing slash |
| `503 reranker_unavailable` / `embedder_unavailable` | OVMS app not ready (Dataset missing, wrong `RAG_OVMS_*_URL`, token rejected); check that app's logs and `GET <OVMS app URL>/v2/health/ready` |
| `504 *_timeout` | raise `RAG_HTTP_TIMEOUT_S` or move the OVMS app to a larger tier |
| `app_ovms_*.sh: OVMS config not found` | Dataset not mounted at `/mnt/data/ovms-models`; set `RAG_OVMS_MODELS_DIR` |
