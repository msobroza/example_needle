# RAG Load Test (Locust, three Domino topologies) Implementation Plan

> **Superseded 2026-09-09:** the file layout in this plan (sub-packages, `scripts/`, `loadtest/locustfile.py`, Chroma) was replaced by the consolidated ten-module layout in §4 of the spec (`docs/superpowers/specs/2026-09-09-rag-load-test-locust-design.md`); the task breakdown below is kept as history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `rag_load_test/` — one FastAPI + LangGraph RAG service whose embedder/reranker transports are switchable between in-process sentence-transformers and OpenVINO Model Server (OVMS) HTTP, plus a Locust suite and comparison script, deployable on Domino 6.2 as a monolith (1 app), split-reranker (2 apps) or split-all (3 apps).

**Architecture:** Hexagonal: `contracts/ports.py` defines async Protocols (`EmbedderPort`, `RerankerPort`, `VectorStorePort`, `ChatModelPort`); adapters implement them; `workflow/graph.py` is a LangGraph `StateGraph` that only sees ports; `api/` wires settings → adapters → graph in a FastAPI lifespan. Locust hits `/query` and `/retrieve`, turns the per-stage `timings_ms` from each response into extra Locust `STAGE` events, and `scripts/compare_runs.py` diffs the resulting CSVs.

**Tech Stack:** Python ≥3.10 (dev on 3.11), FastAPI 0.14x, LangGraph 1.x, chromadb 1.x, sentence-transformers 6.x, openai 3.x (`AsyncOpenAI`), httpx 0.28, pydantic-settings 2.x, locust 2.46, pytest + pytest-asyncio (`asyncio_mode=auto`), ruff + black.

Spec: `docs/superpowers/specs/2026-09-09-rag-load-test-locust-design.md` (read it first).

## Global Constraints

- All new code lives under `rag_load_test/`; it must **not** import `needle` or `needle_core`.
- Python `>=3.10`; type hints on every signature; files ≤ ~300 lines; functions ≤ ~20 lines; nesting ≤ 2 levels (early returns).
- Every network call has a timeout and **bounded** retries (`RAG_HTTP_MAX_RETRIES`, default 2; OpenAI `max_retries` default 2). No circuit breakers / rate limiting.
- Errors carry the offending value + expectation (`RagConfigError`) or component + detail + target (`RagDependencyError`, `kind` ∈ {`unavailable`, `timeout`}).
- Tests never download weights or touch the network; real libraries are exercised only with injected fake model objects, `httpx.MockTransport`, or Chroma `EphemeralClient`.
- `tests/conftest.py` sets `LOCUST_SKIP_MONKEY_PATCH=1` before any `import locust`.
- Lint gate for every task, run from `rag_load_test/`: `ruff check src tests loadtest scripts && black --check src tests loadtest scripts && pytest -q`.
- Dev interpreter for this session: `VENV=/private/tmp/claude-501/-Users-msobroza-Projects-example-needle--claude-worktrees-rag-load-test-locust-bdb5db/a853754a-5c96-4a7b-8ff0-475e66853b89/scratchpad/venv` (`$VENV/bin/python -m pytest`, `$VENV/bin/ruff`, `$VENV/bin/black`). The package is already installed there in editable mode.
- Domino apps bind `0.0.0.0:8888`. OVMS version pinned once in `ovms/versions.env` (`OVMS_VERSION=2026.3.1`).
- Do **not** commit; the user commits. Never add `Co-Authored-By` trailers.
- Model ids: embedder `BAAI/bge-small-en-v1.5` (OVMS servable `bge-small-en-v1.5`), reranker `BAAI/bge-reranker-base` (OVMS servable `bge-reranker-base`), LLM `gpt-4o-mini`.

---

## File structure (ownership per task)

| Task | Creates |
| --- | --- |
| 1 (done) | `pyproject.toml`, `Makefile`, `.env.example`, `src/rag_load_test/{__init__,errors,settings}.py`, `contracts/{models,ports}.py`, `adapters/fake_chat.py`, `testing/fakes.py`, `tests/{conftest,test_settings,test_fakes,test_fake_chat}.py` |
| 2 | `adapters/auth.py`, `adapters/http_retry.py`, `adapters/ovms_embedder.py`, `adapters/ovms_reranker.py`, `tests/test_ovms_adapters.py`, `tests/test_auth.py` |
| 3 | `adapters/executor.py`, `adapters/local_embedder.py`, `adapters/local_reranker.py`, `adapters/chroma_store.py`, `adapters/openai_chat.py`, `tests/test_local_adapters.py`, `tests/test_chroma_store.py`, `tests/test_openai_chat.py` |
| 4 | `workflow/{state,prompts,nodes,graph}.py`, `tests/test_workflow_graph.py` |
| 5 | `ingest/{corpus,ingest_cli}.py`, `tests/test_ingest.py` |
| 6 | `loadtest/{helpers,questions,shapes}.py` (package), `loadtest/locustfile.py` (top-level dir), `scripts/run_loadtest.sh`, `scripts/compare_runs.py`, `tests/test_loadtest_helpers.py`, `tests/test_compare_runs.py`, `tests/test_locustfile_imports.py` |
| 7 | `ovms/{versions.env,export_models.sh,docker-compose.yml,README.md}`, `domino/{app_monolith.sh,app_workflow.sh,app_ovms_reranker.sh,app_ovms_embedder.sh,README.md}`, `domino/environment/{Dockerfile.rag,Dockerfile.ovms}`, `rag_load_test/README.md` |
| 8 | root `.github/workflows/ci.yml` (new job), root `.gitignore`, root `README.md` pointer, root `CHANGELOG.md` |
| 9 (after 2, 3, 4) | `workflow/dependencies.py`, `api/{schemas,tracing,routes,app,__main__}.py`, `tests/test_api.py`, `tests/test_dependencies.py`, `tests/test_tracing.py` |

Tasks 2–8 are independent of each other and only depend on Task 1. Task 9 depends on 2, 3, 4.

---

### Task 1: Foundation (already implemented — read, do not rewrite)

**Files (exist):** see table. Read `src/rag_load_test/contracts/models.py`, `contracts/ports.py`, `errors.py`, `settings.py`, `testing/fakes.py`, `adapters/fake_chat.py` before starting any other task.

**Produces (exact):**

```python
# contracts.models
MetadataValue = str | int | float | bool
RagMode = Literal["query", "retrieve"]
class Passage(BaseModel): id: str; text: str; metadata: dict[str, MetadataValue] = {}
class ScoredPassage(Passage): retrieval_score: float; rerank_score: float | None = None
class StageTimings(BaseModel): embed_ms, retrieve_ms, rerank_ms, generate_ms, total_ms: float = 0.0; def as_dict(self) -> dict[str, float]  # keys embed/retrieve/rerank/generate/total
class RagResult(BaseModel): question: str; mode: RagMode; answer: str | None; passages: list[ScoredPassage]; timings: StageTimings

# contracts.ports (all runtime_checkable Protocols)
EmbedderPort:   model_name: str (property); async embed_query(text) -> list[float]; async embed_documents(texts) -> list[list[float]]; async ready() -> tuple[bool, str]
RerankerPort:   async rerank(query, passages: Sequence[ScoredPassage], top_n) -> list[ScoredPassage]; async ready() -> tuple[bool, str]
VectorStorePort: async upsert(passages, embeddings) -> None; async query(embedding, top_k) -> list[ScoredPassage]; async count() -> int; embedder_model() -> str | None
ChatModelPort:  model_name: str (property); async generate(messages: list[dict[str, str]]) -> str

# errors
RagConfigError(setting, *, got, expected); RagDependencyError(component, detail, *, target=None, kind="unavailable")

# settings
RagSettings(BaseSettings)  # env_prefix RAG_, fields exactly as spec §6; RagSettings.from_env(**overrides); tests use RagSettings(_env_file=None, ...)

# testing
FakeEmbedder(dim=16, name="fake-embedder"); FakeReranker(); InMemoryVectorStore(embedder_model="fake-embedder"); FakeChatModel(latency_ms=0.0) (.calls counter)
```

---

### Task 2: OVMS HTTP adapters + auth + retry

**Files:**
- Create: `src/rag_load_test/adapters/auth.py`
- Create: `src/rag_load_test/adapters/http_retry.py`
- Create: `src/rag_load_test/adapters/ovms_embedder.py`
- Create: `src/rag_load_test/adapters/ovms_reranker.py`
- Test: `tests/test_auth.py`, `tests/test_ovms_adapters.py`

**Interfaces:**
- Consumes: `RagDependencyError`, `ScoredPassage`, `RagSettings` fields `ovms_*`, `http_timeout_s`, `http_max_retries`.
- Produces:

```python
# auth.py
class BearerTokenProvider(Protocol):
    async def token(self) -> str | None: ...
    def invalidate(self) -> None: ...
class NoAuth:                     # token() -> None
class StaticBearerToken:          # __init__(self, token: str)
class DominoAccessToken:
    def __init__(self, client: httpx.AsyncClient, url: str, *, ttl_s: float = 240.0,
                 timeout_s: float = 5.0, clock: Callable[[], float] = time.monotonic) -> None
    # GET url -> body is the raw token (strip whitespace); cached until now-fetched_at >= ttl_s;
    # invalidate() drops the cache; failure -> RagDependencyError("domino_access_token", ..., target=url)
def build_token_provider(settings: RagSettings, client: httpx.AsyncClient) -> BearerTokenProvider
    # "none" -> NoAuth(); "static" -> StaticBearerToken(settings.ovms_bearer_token) (raise RagConfigError("RAG_OVMS_BEARER_TOKEN", got="", expected="non-empty token") if empty); "domino" -> DominoAccessToken(client, settings.domino_access_token_url)

# http_retry.py
async def post_json_with_retry(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], *, component: str,
    token_provider: BearerTokenProvider, timeout_s: float, max_retries: int, backoff_s: float = 0.2,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]
# attempts = max_retries + 1. Retry on httpx.TransportError (incl. TimeoutException) and status >= 500,
# sleeping backoff_s * 2**attempt between attempts. On 401: token_provider.invalidate() and retry (counts as an attempt).
# Other 4xx: raise RagDependencyError(component, f"HTTP {status}: {body[:200]}", target=url) immediately.
# Exhausted: raise RagDependencyError(component, detail, target=url, kind="timeout" if last error was TimeoutException else "unavailable").
# Adds header Authorization: Bearer <token> when token() is not None.

# ovms_embedder.py
class OvmsEmbedder:
    def __init__(self, client: httpx.AsyncClient, base_url: str, model: str, *, token_provider: BearerTokenProvider,
                 timeout_s: float, max_retries: int, batch_size: int = 64) -> None
    model_name -> f"ovms:{model}"
    embed_query(text) -> first vector of embed_documents([text])
    embed_documents(texts) -> POST {base_url}/v3/embeddings {"model": model, "input": [...]} per batch; parse resp["data"], sort by "index", return [d["embedding"]]
    ready() -> GET {base_url}/v3/models/{model} with timeout -> (status==200, f"{url} -> {status}") ; transport error -> (False, str(exc))
# ovms_reranker.py
class OvmsReranker:
    def __init__(self, client, base_url, model, *, token_provider, timeout_s, max_retries) -> None
    rerank(query, passages, top_n) -> POST {base_url}/v3/rerank {"model": model, "query": query, "documents": [p.text ...], "top_n": top_n}
        -> for r in resp["results"]: passages[r["index"]].model_copy(update={"rerank_score": r["relevance_score"]}); sort desc by rerank_score; [:top_n]
        -> empty passages: return [] without calling the server
    ready() -> same shape as OvmsEmbedder.ready
```

- [ ] **Step 1: Write failing tests** — `tests/test_auth.py`:

```python
import httpx, pytest
from rag_load_test.adapters.auth import DominoAccessToken, NoAuth, StaticBearerToken, build_token_provider
from rag_load_test.errors import RagConfigError, RagDependencyError

def _client(handler): return httpx.AsyncClient(transport=httpx.MockTransport(handler))

async def test_static_and_none():
    assert await NoAuth().token() is None
    assert await StaticBearerToken("abc").token() == "abc"

async def test_domino_token_is_cached_until_ttl_and_after_invalidate():
    calls = []
    def handler(req): calls.append(req.url.path); return httpx.Response(200, text="tok-%d\n" % len(calls))
    now = [1000.0]
    provider = DominoAccessToken(_client(handler), "http://localhost:8899/access-token", ttl_s=240, clock=lambda: now[0])
    assert await provider.token() == "tok-1"
    assert await provider.token() == "tok-1"          # cached
    now[0] += 241
    assert await provider.token() == "tok-2"          # ttl expired
    provider.invalidate()
    assert await provider.token() == "tok-3"
    assert calls == ["/access-token"] * 3

async def test_domino_token_failure_is_dependency_error():
    provider = DominoAccessToken(_client(lambda r: httpx.Response(500)), "http://x/access-token")
    with pytest.raises(RagDependencyError) as e: await provider.token()
    assert e.value.component == "domino_access_token" and e.value.target == "http://x/access-token"

def test_build_token_provider_static_requires_token(settings_factory):
    with pytest.raises(RagConfigError) as e:
        build_token_provider(settings_factory(ovms_auth="static", ovms_bearer_token=""), httpx.AsyncClient())
    assert e.value.setting == "RAG_OVMS_BEARER_TOKEN"
```

`tests/test_ovms_adapters.py` (use `httpx.MockTransport`; a handler that records `json.loads(request.content)` and `request.headers`):

```python
async def test_embedder_posts_expected_payload_and_reorders_by_index(): ...  # server returns data out of order [{"index":1,...},{"index":0,...}] -> result[0] is index 0; payload == {"model":"bge-small-en-v1.5","input":["a","b"]}; url path == "/v3/embeddings"
async def test_embedder_batches(): ...  # batch_size=2, 5 texts -> 3 POSTs
async def test_reranker_payload_and_ordering(): ...  # payload has query/documents/top_n; results with relevance_score sorted desc; rerank_score set; retrieval_score preserved; empty passages -> no request
async def test_bearer_header_added_when_token_present(): ...  # StaticBearerToken("t") -> headers["authorization"] == "Bearer t"
async def test_retry_on_503_then_success(): ...  # first 503 then 200; max_retries=2; sleep is a recorded fake; exactly 2 requests; sleep called once with 0.2
async def test_401_invalidates_token_and_retries(): ...  # provider with invalidate counter; first 401 then 200
async def test_4xx_fails_fast(): ...  # 400 -> RagDependencyError kind "unavailable", one request
async def test_timeout_exhausts_to_timeout_kind(): ...  # handler raises httpx.ReadTimeout each time; max_retries=1 -> 2 requests; error.kind == "timeout"; target == url
async def test_ready_true_and_false(): ...  # GET /v3/models/<model> 200 -> (True, ...), ConnectError -> (False, ...)
```

- [ ] **Step 2: Run** `$VENV/bin/python -m pytest tests/test_auth.py tests/test_ovms_adapters.py -q` → fails with ImportError.
- [ ] **Step 3: Implement** the four modules per the interface block. Retry loop shape:

```python
async def post_json_with_retry(...):
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await _post_once(client, url, payload, token_provider, timeout_s)
        except httpx.TransportError as exc:
            last_error = exc
        else:
            if response.status_code == 401:
                token_provider.invalidate(); last_error = _status_error(response)
            elif response.status_code >= 500:
                last_error = _status_error(response)
            elif response.status_code >= 400:
                raise RagDependencyError(component, _describe(response), target=url)
            else:
                return response.json()
        if attempt < max_retries:
            await sleep(backoff_s * (2**attempt))
    kind = "timeout" if isinstance(last_error, httpx.TimeoutException) else "unavailable"
    raise RagDependencyError(component, str(last_error), target=url, kind=kind)
```

- [ ] **Step 4: Run** the two test files → PASS; then the lint gate.

---

### Task 3: Local model adapters, Chroma store, OpenAI chat

**Files:**
- Create: `src/rag_load_test/adapters/executor.py`, `local_embedder.py`, `local_reranker.py`, `chroma_store.py`, `openai_chat.py`
- Test: `tests/test_local_adapters.py`, `tests/test_chroma_store.py`, `tests/test_openai_chat.py`

**Interfaces:**

```python
# executor.py
def build_model_executor(max_workers: int) -> concurrent.futures.ThreadPoolExecutor   # thread_name_prefix="rag-model"
async def run_blocking(executor: Executor, fn: Callable[..., T], *args: Any) -> T       # loop.run_in_executor(executor, functools.partial(fn, *args))

# local_embedder.py
class SentenceTransformerEmbedder:
    def __init__(self, model: Any, executor: Executor, *, model_name: str, batch_size: int = 64) -> None
    @classmethod
    def from_pretrained(cls, model_name: str, executor: Executor, *, batch_size: int = 64) -> "SentenceTransformerEmbedder"
        # imports sentence_transformers INSIDE the method; SentenceTransformer(model_name, device="cpu")
    model_name -> the given name
    embed_documents(texts) -> run_blocking(executor, self._encode, list(texts)) ; _encode calls model.encode(texts, batch_size=..., normalize_embeddings=True, convert_to_numpy=True) and returns [row.tolist() ...]; [] for empty input without calling the model
    embed_query(text) -> (await embed_documents([text]))[0]
    ready() -> (True, model_name)

# local_reranker.py
class CrossEncoderReranker:
    def __init__(self, model: Any, executor: Executor, *, model_name: str, batch_size: int = 32) -> None
    @classmethod
    def from_pretrained(cls, model_name: str, executor: Executor, *, batch_size: int = 32)   # CrossEncoder(model_name, device="cpu") imported inside
    rerank(query, passages, top_n) -> scores = model.predict([(query, p.text) for p in passages], batch_size=...) in executor; float(score) each; model_copy(update={"rerank_score": s}); sort desc; [:top_n]; [] for empty
    ready() -> (True, model_name)

# chroma_store.py
EMBEDDER_MODEL_KEY = "embedder_model"
class ChromaVectorStore:
    def __init__(self, collection: Any, executor: Executor) -> None
    @classmethod
    def open(cls, path: str, collection_name: str, executor: Executor, *, embedder_model: str | None = None) -> "ChromaVectorStore"
        # path == ":memory:" -> chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)) else chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        # get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine", EMBEDDER_MODEL_KEY: embedder_model} (omit key when None))
    upsert(passages, embeddings) -> collection.upsert(ids=[...], documents=[...], embeddings=[list(e) ...], metadatas=[{"passage_id": p.id, **p.metadata} ...]) in executor; no-op for empty
    query(embedding, top_k) -> n = min(top_k, count); if n == 0 return []; collection.query(query_embeddings=[list(embedding)], n_results=n, include=["documents","metadatas","distances"]) -> ScoredPassage(id, text=document, metadata=metadata minus "passage_id", retrieval_score=1.0 - distance)
    count() -> collection.count() in executor
    embedder_model() -> (collection.metadata or {}).get(EMBEDDER_MODEL_KEY)

# openai_chat.py
class OpenAIChatModel:
    def __init__(self, client: Any, model: str, *, temperature: float = 0.0) -> None
    @classmethod
    def from_settings(cls, settings: RagSettings) -> "OpenAIChatModel"   # openai.AsyncOpenAI(timeout=settings.openai_timeout_s, max_retries=settings.openai_max_retries); api_key/base_url come from OPENAI_API_KEY / OPENAI_BASE_URL env (SDK default)
    model_name -> model
    generate(messages) -> resp = await client.chat.completions.create(model=..., messages=..., temperature=...); return resp.choices[0].message.content or ""
        # openai.APITimeoutError -> RagDependencyError("llm", str(exc), target=model, kind="timeout")
        # openai.APIConnectionError / openai.APIStatusError / openai.APIError -> RagDependencyError("llm", ..., target=model)
```

- [ ] **Step 1: Write failing tests.** `tests/test_local_adapters.py` uses fake model objects:

```python
class _FakeSt:
    def __init__(self): self.calls = []
    def encode(self, texts, **kw): self.calls.append((list(texts), kw)); import numpy as np; return np.array([[float(len(t)), 1.0] for t in texts])
class _FakeCe:
    def predict(self, pairs, **kw): import numpy as np; return np.array([float(len(p[1])) for p in pairs])

async def test_st_embedder_normalises_and_returns_lists(): ...   # kw contains normalize_embeddings=True and batch_size; result == [[1.0,1.0],[3.0,1.0]] for ["a","abc"]; empty -> [] and no call
async def test_ce_reranker_orders_and_truncates(): ...           # longer text scores higher -> order; rerank_score float; top_n honoured; empty -> []
async def test_ready_reports_model_name(): ...
```

`tests/test_chroma_store.py` (real chromadb, `:memory:`):

```python
async def test_chroma_round_trip_idempotent_and_ordered(): ...  # open(":memory:", "t", executor, embedder_model="fake-embedder"); upsert 2 passages twice with FakeEmbedder vectors; count()==2; query returns the vacation passage first; metadata restored without passage_id; embedder_model()=="fake-embedder"
async def test_chroma_query_on_empty_collection_returns_empty(): ...
async def test_chroma_query_clamps_top_k(): ...  # 2 rows, top_k=10 -> 2 results
```

`tests/test_openai_chat.py` with a fake client exposing `chat.completions.create` (async) and raising `openai.APITimeoutError(request=httpx.Request("POST","http://x"))` in one test:

```python
async def test_generate_returns_content_and_passes_model_and_temperature(): ...
async def test_timeout_maps_to_timeout_kind(): ...
async def test_status_error_maps_to_unavailable(): ...  # openai.APIStatusError("boom", response=httpx.Response(500, request=httpx.Request("POST","http://x")), body=None)
```

- [ ] **Step 2: Run** → ImportError. **Step 3: Implement.** **Step 4: Run + lint gate.**

---

### Task 4: LangGraph workflow

**Files:**
- Create: `src/rag_load_test/workflow/state.py`, `prompts.py`, `nodes.py`, `graph.py`
- Test: `tests/test_workflow_graph.py`

**Interfaces:**

```python
# state.py
def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]: return {**left, **right}
class RagState(TypedDict, total=False):
    question: str; mode: str; top_k: int; rerank_top_k: int
    query_embedding: list[float]; candidates: list[ScoredPassage]; reranked: list[ScoredPassage]; answer: str | None
    timings_ms: Annotated[dict[str, float], merge_timings]

# prompts.py
SYSTEM_PROMPT: str  # "You answer questions using only the numbered context passages. Cite passages as [n]. If the context is insufficient, say so."
def build_rag_messages(question: str, passages: Sequence[ScoredPassage]) -> list[dict[str, str]]
    # [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"Context:\n[1] ...\n[2] ...\n\nQuestion: ..."}]

# nodes.py  (RagDependencies is defined HERE so Task 9 can import it)
@dataclass
class RagDependencies:
    embedder: EmbedderPort; vector_store: VectorStorePort; reranker: RerankerPort | None; chat_model: ChatModelPort
    top_k_retrieve: int = 20; top_k_rerank: int = 5
NodeFn = Callable[[RagState], Awaitable[dict[str, Any]]]
def timed_node(stage: str, fn: NodeFn) -> NodeFn        # wraps fn; result["timings_ms"] = {stage: elapsed_ms}
def make_embed_node(deps) -> NodeFn      # {"query_embedding": await deps.embedder.embed_query(state["question"])}
def make_retrieve_node(deps) -> NodeFn   # top_k = state.get("top_k") or deps.top_k_retrieve -> {"candidates": [...]}
def make_rerank_node(deps) -> NodeFn     # top_n = state.get("rerank_top_k") or deps.top_k_rerank; reranker None -> passthrough candidates[:top_n]
def make_generate_node(deps) -> NodeFn   # {"answer": await deps.chat_model.generate(build_rag_messages(question, reranked))}

# graph.py
STAGES = ("embed", "retrieve", "rerank", "generate")
def route_after_rerank(state: RagState) -> str      # "generate" if state.get("mode") == "query" else END
def build_rag_graph(deps: RagDependencies) -> CompiledStateGraph   # nodes named "embed_query","retrieve","rerank","generate"
async def run_rag(graph, question: str, *, mode: RagMode = "query", top_k: int | None = None, rerank_top_k: int | None = None) -> RagResult
    # measures total wall time around graph.ainvoke; builds StageTimings(embed_ms=timings.get("embed",0.0), ...) ; answer None in retrieve mode
```

- [ ] **Step 1: Write failing tests** (all with fakes; a `deps` fixture ingests 6 synthetic passages into `InMemoryVectorStore` via `FakeEmbedder`):

```python
async def test_query_mode_runs_all_stages_and_returns_answer(): ...   # answer startswith "[fake-llm]"; timings.embed_ms>=0 etc.; total_ms >= sum of stages - small epsilon? (assert total_ms > 0 and each stage key present)
async def test_retrieve_mode_skips_generate(): ...                    # answer is None; chat.calls == 0; timings.generate_ms == 0.0
async def test_rerank_orders_by_rerank_score_and_truncates(): ...     # rerank_top_k=2 -> 2 passages, rerank_score desc
async def test_reranker_none_passes_top_candidates_through(): ...     # reranker=None -> passages == candidates[:top_n] with rerank_score None
async def test_top_k_override_flows_to_store(): ...                    # spy store records top_k
async def test_timings_merge_reducer(): ...                             # merge_timings({"a":1},{"b":2}) == {"a":1,"b":2}
def test_build_rag_messages_numbers_passages(): ...                     # "[1]" and "[2]" present, question at end
```

- [ ] **Step 2–4:** run (fail) → implement → run (pass) → lint gate. Graph wiring:

```python
builder = StateGraph(RagState)
builder.add_node("embed_query", timed_node("embed", make_embed_node(deps)))
builder.add_node("retrieve", timed_node("retrieve", make_retrieve_node(deps)))
builder.add_node("rerank", timed_node("rerank", make_rerank_node(deps)))
builder.add_node("generate", timed_node("generate", make_generate_node(deps)))
builder.add_edge(START, "embed_query"); builder.add_edge("embed_query", "retrieve"); builder.add_edge("retrieve", "rerank")
builder.add_conditional_edges("rerank", route_after_rerank, {"generate": "generate", END: END})
builder.add_edge("generate", END)
return builder.compile()
```

---

### Task 5: Corpus + ingestion CLI

**Files:**
- Create: `src/rag_load_test/ingest/corpus.py`, `src/rag_load_test/ingest/ingest_cli.py`
- Test: `tests/test_ingest.py`

**Interfaces:**

```python
# corpus.py
class SourceDocument(BaseModel): id: str; title: str; text: str; metadata: dict[str, MetadataValue] = {}
SYNTHETIC_TOPICS: tuple[tuple[str, str, tuple[str, ...]], ...]
  # (topic_slug, title_template, sentence_templates) for a fictional company "Northwind Analytics":
  # vacation policy, expense reimbursement, remote work, security training, onboarding, incident response,
  # data retention, travel booking, performance reviews, equipment requests (10 topics, ≥5 sentences each,
  # every sentence has a {n}/{days}/{amount}-style numeric slot filled from the seeded RNG so documents differ)
def synthetic_corpus(n: int, seed: int = 1234) -> list[SourceDocument]      # deterministic; ids "doc-0000".."doc-{n-1:04d}"; metadata {"topic": slug, "seq": i}
def synthetic_questions(n: int = 50, seed: int = 1234) -> list[str]         # one question template per topic, e.g. "How many vacation days do employees get at Northwind Analytics?"; cycles topics; deterministic
def load_jsonl_corpus(path: Path) -> list[SourceDocument]                   # one JSON object per line: {"id","title","text","metadata"?}; ValueError with line number + offending line[:80] on bad JSON / missing keys
def chunk_document(doc: SourceDocument, *, chunk_words: int = 120, overlap_words: int = 20) -> list[Passage]
  # sliding word windows; ids f"{doc.id}:{i}"; metadata {**doc.metadata, "doc_id": doc.id, "title": doc.title, "chunk": i}; overlap < chunk required (ValueError with values)

# ingest_cli.py
async def ingest_passages(passages: Sequence[Passage], embedder: EmbedderPort, store: VectorStorePort, *, batch_size: int = 64) -> int   # returns number of passages upserted
def build_parser() -> argparse.ArgumentParser   # --synthetic N (default 200) | --corpus PATH ; --seed 1234 ; --chunk-words 120 ; --overlap-words 20 ; --batch-size 64 ; --fake (use FakeEmbedder + InMemoryVectorStore, for tests/demos)
def main(argv: Sequence[str] | None = None) -> int
  # settings = RagSettings.from_env(); executor = build_model_executor(settings.model_threads)
  # embedder: settings.embedder_backend == "fake" or --fake -> FakeEmbedder(); "local" -> SentenceTransformerEmbedder.from_pretrained; "ovms" -> OvmsEmbedder (with httpx.AsyncClient + build_token_provider)
  # store: --fake -> InMemoryVectorStore() else ChromaVectorStore.open(settings.chroma_path, settings.chroma_collection, executor, embedder_model=embedder.model_name)
  # prints one JSON line {"event":"ingest_done","documents":..,"passages":..,"collection":..,"embedder":..} ; returns 0
```

- [ ] **Step 1: Failing tests:** determinism (`synthetic_corpus(5) == synthetic_corpus(5)`, differs with seed), question count/cycle, chunking (word counts, overlap, ids, ValueError when overlap ≥ chunk), JSONL loader (good file via `tmp_path`, bad line raises with line number), `ingest_passages` with fakes returns count and store count matches, `main(["--synthetic","3","--fake"])` returns 0 and prints the JSON line (capsys).
- [ ] **Step 2–4:** fail → implement → pass → lint gate.

---

### Task 6: Locust suite + comparison script

**Files:**
- Create: `src/rag_load_test/loadtest/helpers.py`, `questions.py`, `shapes.py`
- Create: `loadtest/locustfile.py` (top-level `rag_load_test/loadtest/`, NOT the package)
- Create: `scripts/run_loadtest.sh` (chmod +x), `scripts/compare_runs.py`
- Test: `tests/test_loadtest_helpers.py`, `tests/test_compare_runs.py`, `tests/test_locustfile_imports.py`

**Interfaces:**

```python
# helpers.py (no locust import)
STAGE_REQUEST_TYPE = "STAGE"
def auth_headers(env: Mapping[str, str]) -> dict[str, str]
  # RAG_LOADTEST_BEARER_TOKEN -> {"Authorization": "Bearer <t>"}; elif DOMINO_API_KEY -> {"X-Domino-Api-Key": k}; else {}
def stage_events(endpoint: str, timings_ms: Mapping[str, float]) -> list[tuple[str, float]]
  # [(f"{endpoint}:{stage}", ms) for stage in ("embed","retrieve","rerank","generate") if stage in timings_ms and ms > 0] — total excluded
def task_weights(env: Mapping[str, str]) -> tuple[int, int]   # (RAG_LOADTEST_QUERY_WEIGHT default 1, RAG_LOADTEST_RETRIEVE_WEIGHT default 3); ValueError with value on non-int/negative
def pick_question(rng: random.Random, questions: Sequence[str]) -> str
def parse_timings(body: Mapping[str, Any]) -> dict[str, float]   # body["timings_ms"] with float values; {} when missing/malformed

# questions.py
QUESTIONS: list[str] = synthetic_questions(50)   # from rag_load_test.ingest.corpus

# shapes.py (imports locust)
STEP_USERS = (5, 10, 20, 40); STEP_SECONDS = 60
class StepLoadShape(LoadTestShape):
    def tick(self) -> tuple[int, float] | None   # step index = int(run_time // STEP_SECONDS); None after the last step; spawn rate = users of the step

# loadtest/locustfile.py
import os, random
from locust import FastHttpUser, between, task
from rag_load_test.loadtest.helpers import ...; from rag_load_test.loadtest.questions import QUESTIONS
QUERY_WEIGHT, RETRIEVE_WEIGHT = task_weights(os.environ)
class RagUser(FastHttpUser):
    wait_time = between(0.2, 1.0)
    def on_start(self): self.rng = random.Random(); self.headers = auth_headers(os.environ)
    @task(QUERY_WEIGHT) def query(self): self._ask("/query")
    @task(RETRIEVE_WEIGHT) def retrieve(self): self._ask("/retrieve")
    def _ask(self, path):
        payload = {"question": pick_question(self.rng, QUESTIONS)}
        with self.client.post(path, json=payload, headers=self.headers, name=path, catch_response=True) as resp:
            if resp.status_code != 200: resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}"); return
            try: body = resp.json()
            except ValueError: resp.failure("non-JSON body"); return
            for name, ms in stage_events(path, parse_timings(body)):
                self.environment.events.request.fire(request_type=STAGE_REQUEST_TYPE, name=name, response_time=ms, response_length=0, exception=None, context={})
if os.environ.get("RAG_LOADTEST_SHAPE") == "step":
    from rag_load_test.loadtest.shapes import StepLoadShape  # noqa: F401  (locust auto-registers shape classes found in the locustfile module)

# scripts/compare_runs.py (stdlib only)
@dataclass(frozen=True)
class RowStats: name: str; request_type: str; requests: int; failures: int; rps: float; p50: float; p95: float; p99: float
def load_stats(prefix: Path) -> dict[str, RowStats]          # reads f"{prefix}_stats.csv"; key f"{Type} {Name}"; skips the "Aggregated" row into key "Aggregated"; columns by header name: "Request Count","Failure Count","Requests/s","50%","95%","99%"
def render_markdown(runs: dict[str, dict[str, RowStats]]) -> str
  # one table per row-key present in any run; columns: run, requests, fail %, RPS, p50, p95, p99, Δp95 vs first run (ms and %); rows sorted with endpoints (POST /query, POST /retrieve) first, then STAGE rows, then Aggregated
def main(argv: Sequence[str] | None = None) -> int   # positional prefixes..., --out PATH (default: print to stdout); run label = prefix basename
```

`scripts/run_loadtest.sh`:

```bash
#!/usr/bin/env bash
# Usage: scripts/run_loadtest.sh <run-name> <host> [users=20] [spawn-rate=5] [run-time=2m]
set -euo pipefail
NAME=${1:?run name}; HOST=${2:?host url}; USERS=${3:-20}; SPAWN=${4:-5}; TIME=${5:-2m}
cd "$(dirname "$0")/.."
mkdir -p results
locust -f loadtest/locustfile.py --headless --host "$HOST" -u "$USERS" -r "$SPAWN" -t "$TIME" \
  --csv "results/$NAME" --html "results/$NAME.html" --only-summary
echo "results/${NAME}_stats.csv"
```

- [ ] **Step 1: Failing tests:** helpers (each function, including weights errors), `render_markdown` on two fixture CSVs written to `tmp_path` (assert the `POST /query` table has both runs and the Δp95 column, and that a `STAGE /query:rerank` table exists), `main([...,"--out",path])` writes the file, and `test_locustfile_imports.py` imports `loadtest/locustfile.py` via `importlib.util.spec_from_file_location` with `LOCUST_SKIP_MONKEY_PATCH=1` and asserts `RagUser.tasks` has 2 entries and `StepLoadShape.tick()` returns `(5, 5)` at t=0 and `None` after 240 s (use a fake `get_run_time`).
- [ ] **Step 2–4:** fail → implement → pass → lint gate (ruff/black also cover `loadtest/` and `scripts/`).

---

### Task 7: OVMS assets, Domino entry points, docs

**Files:**
- Create: `ovms/versions.env`, `ovms/export_models.sh` (+x), `ovms/docker-compose.yml`, `ovms/README.md`
- Create: `domino/app_monolith.sh`, `domino/app_workflow.sh`, `domino/app_ovms_reranker.sh`, `domino/app_ovms_embedder.sh` (all +x), `domino/environment/Dockerfile.rag`, `domino/environment/Dockerfile.ovms`, `domino/README.md`
- Create: `rag_load_test/README.md`

**Content requirements (verbatim values):**

`ovms/versions.env`:
```
OVMS_VERSION=2026.3.1
OVMS_IMAGE=openvino/model_server:2026.3.1
OVMS_BINARY_URL=https://github.com/openvinotoolkit/model_server/releases/download/v2026.3.1/ovms_ubuntu22_2026.3.1_python_off.tar.gz
RERANKER_SOURCE_MODEL=BAAI/bge-reranker-base
RERANKER_MODEL_NAME=bge-reranker-base
EMBEDDER_SOURCE_MODEL=BAAI/bge-small-en-v1.5
EMBEDDER_MODEL_NAME=bge-small-en-v1.5
```

`ovms/export_models.sh`: sources `versions.env`; `MODELS_DIR=${1:-$(dirname "$0")/models}`; downloads `https://raw.githubusercontent.com/openvinotoolkit/model_server/v${OVMS_VERSION}/demos/common/export_models/export_model.py` and the sibling `requirements.txt` into `$MODELS_DIR/.export/` (skips if present); `pip install -r requirements.txt`; runs
```
python export_model.py rerank_ov --source_model "$RERANKER_SOURCE_MODEL" --model_name "$RERANKER_MODEL_NAME" --weight-format int8 --config_file_path "$MODELS_DIR/config_reranker.json" --model_repository_path "$MODELS_DIR"
python export_model.py embeddings_ov --source_model "$EMBEDDER_SOURCE_MODEL" --model_name "$EMBEDDER_MODEL_NAME" --pooling CLS --weight-format int8 --config_file_path "$MODELS_DIR/config_embedder.json" --model_repository_path "$MODELS_DIR"
```
and prints the two config paths.

`ovms/docker-compose.yml`: services `reranker` (`${OVMS_IMAGE}`, ports `8001:8001`, volume `./models:/workspace:ro`, command `--rest_port 8001 --config_path /workspace/config_reranker.json`) and `embedder` (ports `8002:8002`, command `--rest_port 8002 --config_path /workspace/config_embedder.json`).

Domino scripts (all): `#!/usr/bin/env bash`, `set -euo pipefail`, `cd "$(dirname "$0")/.."`, `export RAG_HOST=0.0.0.0 RAG_PORT=8888`, and:
- `app_monolith.sh`: `export RAG_TOPOLOGY=${RAG_TOPOLOGY:-monolith} RAG_EMBEDDER_BACKEND=local RAG_RERANKER_BACKEND=local`; `exec rag-serve`.
- `app_workflow.sh`: requires `RAG_OVMS_RERANK_URL`; `RAG_RERANKER_BACKEND=ovms`; `RAG_EMBEDDER_BACKEND=${RAG_EMBEDDER_BACKEND:-local}` (set to `ovms` + `RAG_OVMS_EMBEDDINGS_URL` for split-all); `RAG_OVMS_AUTH=${RAG_OVMS_AUTH:-domino}`; `RAG_TOPOLOGY` default derived: `split-all` if embedder ovms else `split-reranker`; `exec rag-serve`.
- `app_ovms_reranker.sh` / `app_ovms_embedder.sh`: `MODELS_DIR=${RAG_OVMS_MODELS_DIR:-/mnt/data/ovms-models}`; `exec ovms --rest_port 8888 --config_path "$MODELS_DIR/config_reranker.json"` (resp. `config_embedder.json`); fail with a clear message if the config file is missing.

`domino/environment/Dockerfile.rag`: `FROM` a Domino Standard Environment (`ARG BASE_IMAGE=quay.io/domino/compute-environment-images:ubuntu22-py3.10-r4.4-domino6.2-standard`), `COPY rag_load_test /opt/rag_load_test`, `pip install /opt/rag_load_test[loadtest,domino]`, pre-download the two HF models with `python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"`, `ENV HF_HUB_OFFLINE=1`.
`Dockerfile.ovms`: same base, `curl -L $OVMS_BINARY_URL | tar -xz -C /opt`, `ENV PATH=/opt/ovms/bin:$PATH LD_LIBRARY_PATH=/opt/ovms/lib`.

`domino/README.md` must contain: a table of the three topologies → apps → entry point → env vars; how to publish an App (Apps view → entry point script, hardware tier ≥4 CPU cores) and how to deploy as an Agent (Experiment Manager → Deploy Agent, `RAG_DOMINO_TRACING=true`, `[domino]` extra); how to load the exported models into a Domino Dataset mounted at `/mnt/data/ovms-models`; app-to-app auth (`RAG_OVMS_AUTH=domino` → `http://localhost:8899/access-token`); calling the apps from Locust inside Domino (bearer token in `RAG_LOADTEST_BEARER_TOKEN`) or from a laptop (`DOMINO_API_KEY` → `X-Domino-Api-Key`); the `DOMINO_RUN_HOST_PATH` proxy note.

`rag_load_test/README.md`: Mermaid diagram of the workflow + three topology diagrams, quickstart (`make install-dev`, `make ingest`, `make serve`, `curl -X POST localhost:8888/query -d '{"question":"..."}'`), local split topologies via `make ovms-export && make ovms-up` + env vars, load test commands (`make loadtest RUN_NAME=monolith HOST=...`, three runs, `make compare`), settings table (copy from spec §6), test/lint commands.

- [ ] **Step 1:** write the files. **Step 2:** `bash -n` every `.sh`; `docker compose -f ovms/docker-compose.yml --env-file ovms/versions.env config` if docker is available (otherwise skip and say so). **Step 3:** grep the READMEs for `TODO|TBD|XXX` → none.

---

### Task 8: Root repository integration

**Files:**
- Modify: `.github/workflows/ci.yml` (append a job), `.gitignore` (append), `README.md` (add section before "## Development"), `CHANGELOG.md` (Unreleased → Added).

`ci.yml` job:
```yaml
  rag-load-test:
    name: rag_load_test (py3.11)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: rag_load_test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install rag_load_test + dev tooling
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"
      - name: Ruff
        run: ruff check src tests loadtest scripts
      - name: Black
        run: black --check src tests loadtest scripts
      - name: Pytest
        run: pytest -q
```
`.gitignore` additions: `rag_load_test/data/chroma/`, `rag_load_test/results/*`, `!rag_load_test/results/.gitkeep`, `rag_load_test/ovms/models/`, `rag_load_test/.env`.
README section "## Load testing the RAG workflow (`rag_load_test/`)" — 6–10 lines + link. CHANGELOG bullet under Unreleased/Added.

- [ ] **Step 1:** apply edits. **Step 2:** `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` (or `$VENV/bin/python`) → OK; root `make lint` still passes (`$VENV/bin/ruff check src tests examples scripts` from repo root).

---

### Task 9: Dependency wiring + FastAPI app (after Tasks 2, 3, 4)

**Files:**
- Create: `src/rag_load_test/workflow/dependencies.py`, `src/rag_load_test/api/schemas.py`, `api/tracing.py`, `api/routes.py`, `api/app.py`, `api/__main__.py`
- Test: `tests/test_dependencies.py`, `tests/test_tracing.py`, `tests/test_api.py`

**Interfaces:**

```python
# workflow/dependencies.py
def build_embedder(settings, *, http_client, executor, token_provider) -> EmbedderPort     # local -> SentenceTransformerEmbedder.from_pretrained(settings.embedder_model, executor); ovms -> OvmsEmbedder(...); fake -> FakeEmbedder()
def build_reranker(settings, *, http_client, executor, token_provider) -> RerankerPort | None   # none -> None
def build_vector_store(settings, *, executor, embedder_model: str) -> VectorStorePort         # chroma_path ":memory:" allowed; embedder_model recorded
def build_chat_model(settings) -> ChatModelPort   # openai -> OpenAIChatModel.from_settings; fake -> FakeChatModel(settings.fake_llm_latency_ms)
def build_dependencies(settings, *, http_client: httpx.AsyncClient, executor: Executor) -> RagDependencies
   # token_provider = build_token_provider(settings, http_client); passes top_k_retrieve/top_k_rerank

# api/schemas.py
class QueryRequest(BaseModel): question: str = Field(min_length=1, max_length=2000); top_k: int | None = Field(None, ge=1, le=100); rerank_top_k: int | None = Field(None, ge=1, le=50)
class PassageOut(BaseModel): id: str; text: str; retrieval_score: float; rerank_score: float | None; metadata: dict[str, MetadataValue]
class QueryResponse(BaseModel): request_id: str; deployment: str; mode: RagMode; answer: str | None; passages: list[PassageOut]; timings_ms: dict[str, float]
class ReadyCheck(BaseModel): name: str; ok: bool; detail: str
class ReadyResponse(BaseModel): status: Literal["ready", "not_ready"]; checks: list[ReadyCheck]
def to_query_response(result: RagResult, *, request_id: str, deployment: str) -> QueryResponse   # timings_ms = result.timings.as_dict()

# api/tracing.py
def traced(name: str, *, enabled: bool) -> Callable[[F], F]
   # enabled False -> identity. enabled True -> try `from domino.aisystems.tracing import add_tracing` then `from domino.agents.tracing import add_tracing`; apply add_tracing(name=name, autolog_frameworks=["langchain"]); ImportError -> log a warning (json) and return identity.
   # exposes `resolve_add_tracing() -> Callable | None` for tests (monkeypatchable)

# api/app.py
@dataclass
class RagRuntime: settings: RagSettings; deps: RagDependencies; graph: Any
async def readiness_checks(runtime: RagRuntime) -> list[ReadyCheck]
   # "vector_store": count > 0 (detail f"{n} passages in {collection}"); "embedder": ready(); "reranker": ready() when not None; "embedder_model_match": ok always, detail warns when store.embedder_model() not in (None, deps.embedder.model_name)
def create_app(settings: RagSettings | None = None, *, dependencies: RagDependencies | None = None) -> FastAPI
   # lifespan: executor = build_model_executor(settings.model_threads); http_client = httpx.AsyncClient(timeout=settings.http_timeout_s); deps = dependencies or build_dependencies(...); graph = build_rag_graph(deps); app.state.runtime = RagRuntime(...); on shutdown: await http_client.aclose(); executor.shutdown(wait=False)
   # FastAPI(title="rag-load-test", root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""), lifespan=lifespan); include routes.router
   # exception handlers: RagDependencyError -> 504 if kind == "timeout" else 503, body {"error": f"{component}_{kind}", "detail": str(exc), "target": target}
   # middleware: X-Request-ID (uuid4 hex if absent) and X-RAG-Topology on every response
def build_app_from_env() -> FastAPI   # create_app(RagSettings.from_env())

# api/routes.py
router = APIRouter()
POST /query   -> run_rag(runtime.graph, body.question, mode="query", ...) wrapped by traced("rag_query", enabled=settings.domino_tracing)
POST /retrieve -> mode="retrieve"
GET /healthz  -> {"status": "ok"}
GET /readyz   -> ReadyResponse; HTTP 503 when any check not ok
# each handler logs one JSON line: {"event":"rag_request","request_id":..,"topology":..,"mode":..,"timings_ms":{...},"status":200}

# api/__main__.py
def main(argv: Sequence[str] | None = None) -> int   # --host/--port override settings; uvicorn.run(create_app(settings), host=..., port=..., log_level=settings.log_level)
```

- [ ] **Step 1: Failing tests.** `tests/test_api.py` builds `create_app(settings_factory(topology="test-topo", reranker_backend="fake", embedder_backend="fake", llm_backend="fake", chroma_path=":memory:"), dependencies=deps)` with fake deps pre-populated (6 passages) and uses `fastapi.testclient.TestClient(app)` as a context manager (runs lifespan):

```python
def test_query_returns_answer_passages_timings_and_headers(): ...   # 200; body.answer startswith "[fake-llm]"; len(passages) <= 5; timings_ms keys embed/retrieve/rerank/generate/total; headers X-RAG-Topology == "test-topo"; X-Request-ID present; deployment == "test-topo"
def test_retrieve_has_no_answer(): ...                                 # answer None; timings_ms["generate"] == 0.0
def test_validation_422_on_empty_question(): ...
def test_healthz(): ...
def test_readyz_ready(): ...                                           # 200, status "ready", check names include vector_store/embedder/reranker
def test_readyz_not_ready_when_store_empty(): ...                      # empty InMemoryVectorStore -> 503, checks[vector_store].ok False
def test_dependency_error_maps_to_503_and_timeout_to_504(): ...        # reranker fake that raises RagDependencyError(kind="unavailable"/"timeout"); body.error == "reranker_unavailable" / "reranker_timeout"
def test_request_id_is_propagated(): ...                               # send X-Request-ID: abc -> response header abc and body.request_id abc
```

`tests/test_dependencies.py`: with `fake` backends + `chroma_path=":memory:"`, `build_dependencies` returns FakeEmbedder/FakeReranker/ChromaVectorStore/FakeChatModel; `reranker_backend="none"` → `None`; `embedder_backend="ovms"` → `OvmsEmbedder` whose `model_name == "ovms:bge-small-en-v1.5"` (no network happens at construction); `llm_backend="openai"` with `OPENAI_API_KEY=test` env → `OpenAIChatModel` (construction only).
`tests/test_tracing.py`: `traced("x", enabled=False)` returns the same function object; `enabled=True` with `resolve_add_tracing` monkeypatched to a recording decorator applies it with `name="x"` and `autolog_frameworks=["langchain"]`; `enabled=True` with resolver returning `None` → identity.

- [ ] **Step 2–4:** fail → implement → pass → full lint gate over the whole sub-project (`ruff check src tests loadtest scripts && black --check src tests loadtest scripts && pytest -q`).

---

## Self-review

- **Spec coverage:** §4 layout → Tasks 1–9 (every file listed has an owner); §5 ports → Task 1; §6 settings → Task 1; §7 workflow → Task 4; §8 API → Task 9; §9 ingestion → Task 5; §10 Locust → Task 6; §11 OVMS → Task 7; §12 Domino → Task 7; §13 robustness → Tasks 2, 3, 9; §14 tests → each task; §15 root integration → Task 8.
- **Placeholders:** none; every value that must be exact is written out (model ids, ports, env names, OVMS flags, CSV columns).
- **Type consistency:** `RagDependencies` is defined in `workflow/nodes.py` (Task 4) and imported by Task 9; `run_rag` returns `RagResult` consumed by `to_query_response`; `stage_events` reads keys produced by `StageTimings.as_dict()`; `OvmsEmbedder.model_name` is `ovms:<model>` in both Task 2 and Task 9's test.
