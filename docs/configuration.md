# Configuration

`needle.config.RetrieverConfig` bundles the knobs needed to construct a
model-backed retriever. It can be populated from the environment, making it easy
to drive the library from a CLI, a service or a notebook without hard-coding
values. Once you have a config, `needle.factory.build_retriever` turns it into a
concrete retriever.

```python
from needle.config import RetrieverConfig
from needle.factory import build_retriever
```

## `RetrieverConfig`

`RetrieverConfig` is a dataclass with the following fields and defaults:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | `"colqwen2"` | Retriever backend name (see [Retrievers](retrievers.md)). |
| `index_path` | `str` | `"index.pkl"` (`DEFAULT_INDEX_PATH`) | Where the on-disk index lives. |
| `dpi` | `int` | `150` (`DEFAULT_DPI`) | Rasterisation resolution (see [Extractors](extractors.md)). |
| `batch_size` | `int` | `4` (`DEFAULT_BATCH_SIZE`) | Page images embedded per forward pass. |
| `model_name` | `Optional[str]` | `None` | Override the backend's default model id / path. |

The defaults come from `needle.constants`.

### Validation

`__post_init__` validates the config and raises
`needle.exceptions.ConfigurationError` on invalid input:

* `name` must be non-empty.
* `dpi` must be positive.
* `batch_size` must be positive.

```python
from needle.config import RetrieverConfig
from needle.exceptions import ConfigurationError

try:
    RetrieverConfig(dpi=0)
except ConfigurationError as exc:
    print(exc)   # dpi must be positive, got 0
```

### `to_kwargs()`

`to_kwargs()` returns the keyword arguments for
`get_retriever(name, **kwargs)`. It always includes `index_path`, `dpi` and
`batch_size`, and adds `model_name` **only when it is set** (so the backend's own
default model id is used otherwise):

```python
config = RetrieverConfig(name="colqwen2", dpi=200)
config.to_kwargs()
# {'index_path': 'index.pkl', 'dpi': 200, 'batch_size': 4}

RetrieverConfig(name="colpali", model_name="vidore/colpali").to_kwargs()
# {'index_path': 'index.pkl', 'dpi': 150, 'batch_size': 4, 'model_name': 'vidore/colpali'}
```

### `from_env()`

`RetrieverConfig.from_env(prefix="NEEDLE_")` reads the config from environment
variables, falling back to the field defaults when a variable is unset:

| Variable | Field | Parsed as |
| --- | --- | --- |
| `NEEDLE_RETRIEVER` | `name` | `str` |
| `NEEDLE_INDEX_PATH` | `index_path` | `str` |
| `NEEDLE_DPI` | `dpi` | `int` |
| `NEEDLE_BATCH_SIZE` | `batch_size` | `int` |
| `NEEDLE_MODEL_NAME` | `model_name` | `str` (or `None` if empty/unset) |

```python
import os
from needle.config import RetrieverConfig

os.environ["NEEDLE_RETRIEVER"] = "colpali"
os.environ["NEEDLE_DPI"] = "200"

config = RetrieverConfig.from_env()
# RetrieverConfig(name='colpali', index_path='index.pkl', dpi=200, batch_size=4, model_name=None)
```

Pass a different `prefix` to namespace the variables differently
(`RetrieverConfig.from_env(prefix="MYAPP_")`).

## Building a retriever from a config

`needle.factory.build_retriever(config)` instantiates a concrete retriever by
calling `get_retriever(config.name, **config.to_kwargs())`. It imports the
registry lazily, so **torch is only pulled in when this is called** — importing
`needle.factory` itself stays cheap.

```python
from needle.config import RetrieverConfig
from needle.factory import build_retriever

config = RetrieverConfig(name="colqwen2", index_path="my_index.pkl", dpi=200)
retriever = build_retriever(config)   # imports torch + the model backend
```

!!! note "Real backends need the `[retrieval]` extra"
    `build_retriever` constructs a model-backed retriever, which requires the
    optional `torch` + `colpali-engine` dependencies. Install them with
    `pip install -e ".[retrieval]"`. For a fully torch-free alternative, use
    [`build_pipeline`](pipeline.md) instead.

## Real model backends

The model + processor pairs the concrete retrievers load are centralised in
`needle.embedders.colpali_engine`. Importing the module is cheap: `torch` and
`colpali_engine` are only imported **inside** the loader functions, so nothing
here requires a GPU or model weights at import time.

### `BACKEND_SPECS`

`BACKEND_SPECS` maps a logical backend name to a `BackendSpec`, which records the
names of the `colpali_engine` model and processor classes plus whether the backend
is multi-vector:

| Backend | Model class | Processor class | `multi_vector` |
| --- | --- | --- | --- |
| `colpali` | `ColPali` | `ColPaliProcessor` | `True` |
| `colqwen2` | `ColQwen2` | `ColQwen2Processor` | `True` |
| `colmodernvbert` | `ColModernVBert` | `ColModernVBertProcessor` | `True` |
| `nomic` | `BiIdefics3` | `BiIdefics3Processor` | `False` |
| `modernvbert` | `BiModernVBert` | `BiModernVBertProcessor` | `False` |

```python
from needle.embedders.colpali_engine import BACKEND_SPECS

spec = BACKEND_SPECS["colqwen2"]
print(spec.model_cls, spec.processor_cls, spec.multi_vector)
# ColQwen2 ColQwen2Processor True
```

### `load_backend()` and `resolve_device()`

`load_backend(backend, model_name, *, device=None)` returns a
`(model, processor)` tuple for a named backend, loaded from `model_name`. It
requires the `[retrieval]` extra. Unknown backend names raise `KeyError`.

```python
from needle.embedders.colpali_engine import load_backend

model, processor = load_backend("colqwen2", "vidore/colqwen2-v1.0-hf")
```

`resolve_device(device=None)` returns an explicit device, defaulting to `"cuda"`
when a CUDA-capable GPU is available and `"cpu"` otherwise (and `"cpu"` if torch is
not installed). `load_backend` uses it to pick `torch.bfloat16` on CUDA and
`torch.float32` on CPU.
