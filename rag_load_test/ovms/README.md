# OpenVINO Model Server assets

Everything needed to serve the reranker and the embedder over HTTP with
[OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) (OVMS),
either locally with docker compose or on Domino (`domino/app_ovms_*.sh`).

| File | Purpose |
| --- | --- |
| `versions.env` | Single pin for the OVMS version, image and binary tarball, plus the two model ids. Sourced by `export_models.sh`; passed to docker compose with `--env-file`. |
| `export_models.sh` | Exports both models to OpenVINO IR (int8) into an OVMS model repository (`ovms/models/` by default, git-ignored). |
| `docker-compose.yml` | Two OVMS containers: reranker on port `8001`, embedder on port `8002`. |

Pinned values (`versions.env`):

| Key | Value |
| --- | --- |
| `OVMS_VERSION` | `2026.3.1` |
| `OVMS_IMAGE` | `openvino/model_server:2026.3.1` |
| `OVMS_BINARY_URL` | `ovms_ubuntu22_2026.3.1_python_off.tar.gz` from the GitHub release `v2026.3.1` |
| `RERANKER_SOURCE_MODEL` / `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` / `bge-reranker-base` |
| `EMBEDDER_SOURCE_MODEL` / `EMBEDDER_MODEL_NAME` | `BAAI/bge-small-en-v1.5` / `bge-small-en-v1.5` |

The `*_MODEL_NAME` values are the servable names OVMS exposes; they are the
defaults of `RAG_OVMS_RERANK_MODEL` and `RAG_OVMS_EMBEDDINGS_MODEL`.

## 1. Export the models (once, needs HF Hub access)

```bash
make ovms-export                 # == ./ovms/export_models.sh [MODELS_DIR]
```

The script:

1. Downloads `export_model.py` and its `requirements.txt` from the OVMS repository
   at tag `v${OVMS_VERSION}` into `ovms/models/.export/` (cached; delete that
   directory to refresh them).
2. Runs `pip install -r requirements.txt` (optimum-intel, openvino, ...). Use a
   dedicated virtualenv (`PYTHON=/path/to/venv/bin/python make ovms-export`) if
   you do not want those packages next to the service.
3. Runs the two exports with `--weight-format int8`:
   `rerank_ov` for the reranker and `embeddings_ov --pooling CLS` for the embedder.

Result:

```text
ovms/models/
  config_reranker.json        # --config_path for the reranker server
  config_embedder.json        # --config_path for the embedder server
  bge-reranker-base/          # OpenVINO IR + tokenizer graph
  bge-small-en-v1.5/
  .export/                    # cached exporter script + requirements
```

The config files reference the model directories relative to their own
location, so the whole directory can be mounted anywhere: `/workspace` in the
compose file, `/mnt/data/ovms-models` on Domino (see `../domino/README.md`).
Exporting straight into a Domino Dataset works too:
`./ovms/export_models.sh /mnt/data/ovms-models` from a Workspace.

## 2. Run locally

```bash
make ovms-up      # docker compose -f ovms/docker-compose.yml --env-file ovms/versions.env up -d
make ovms-down
```

Verify both servers (health, then one real call each):

```bash
curl -s localhost:8001/v2/health/ready && echo reranker ready
curl -s localhost:8002/v2/health/ready && echo embedder ready

curl -s localhost:8001/v3/rerank -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-base","query":"vacation days","documents":["Employees get 25 vacation days.","Expense reports are due monthly."],"top_n":2}'

curl -s localhost:8002/v3/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"bge-small-en-v1.5","input":["vacation policy"]}'
```

## 3. Point the service at OVMS

```bash
# split-reranker: reranker over HTTP, embedder in-process
RAG_TOPOLOGY=split-reranker RAG_RERANKER_BACKEND=ovms RAG_OVMS_RERANK_URL=http://localhost:8001 make serve

# split-all: both over HTTP
RAG_TOPOLOGY=split-all RAG_RERANKER_BACKEND=ovms RAG_OVMS_RERANK_URL=http://localhost:8001 \
  RAG_EMBEDDER_BACKEND=ovms RAG_OVMS_EMBEDDINGS_URL=http://localhost:8002 make serve
```

`RAG_OVMS_AUTH` stays `none` locally; on Domino the workflow app uses `domino`
(bearer token from the run) or `static` (`RAG_OVMS_BEARER_TOKEN`).

## 4. API contract used by the adapters

| Adapter | Request | Response |
| --- | --- | --- |
| `OvmsReranker` → `POST {RAG_OVMS_RERANK_URL}/v3/rerank` | `{"model","query","documents":[...],"top_n"}` | `{"results":[{"index","relevance_score"}]}` |
| `OvmsEmbedder` → `POST {RAG_OVMS_EMBEDDINGS_URL}/v3/embeddings` | `{"model","input":[...]}` | `{"data":[{"index","embedding":[...]}]}` |

Both adapters re-order results by `index`, use `RAG_HTTP_TIMEOUT_S` per request
and retry at most `RAG_HTTP_MAX_RETRIES` times (default 2, backoff 0.2 s, 0.4 s)
on connection errors and 5xx. Base URLs must not end with a slash.

## 5. Bumping OVMS

Change `versions.env` only, re-run `make ovms-export` (the exporter is fetched
from the new tag), and rebuild the Domino environment from
`domino/environment/Dockerfile.ovms` with `--build-arg OVMS_BINARY_URL=...`
taken from the same file.
