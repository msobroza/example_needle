"""Page-level retrievers — multi-format, multi-model.

Built-in multi-vectors and embedding.
Add your own by subclassing BaseRetriever.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

import numpy as np
import torch
from needle_core.domain.interaction.query import Query
from needle_core.domain.metadata.backend_filter_adapters import (
    MultiFieldFilterAdapter,
)
from needle_core.domain.types import DocumentExtension
from tqdm import tqdm

from .data import InputDocument, PageAnnotationResult, PreannotationPageResult
from .extractors import IMAGE_EXTRACTORS, PageExtractor, PageToImageExtractor
from .page_retriever_utils import SimilarityMapVisualizer, matches_filter

logger = logging.getLogger(__name__)


class BasePageRetriever(ABC):
    """Pure abstract base class for all retrievers in the pipeline."""

    @property
    def supported_extensions(self) -> tuple[DocumentExtension]:
        "Get supported extensions"
        return tuple(self.page_extractors.keys())

    @abstractproperty
    def page_extractors(self) -> dict[DocumentExtension, PageExtractor]:
        "Available page extractors per extension"

    @abstractmethod
    def save_index(self, index_path: Path, **kwargs) -> None:
        "Define function to persist indexes"

    @abstractmethod
    def load_index(self, **kwargs) -> BasePageRetriever:
        "Define function to load indexes"

    @abstractmethod
    def reinit_indexes(self) -> BasePageRetriever:
        """Re-init indexes"""

    @abstractmethod
    def search(self, query: Query, top_k: int = 10) -> list[PageAnnotationResult]:
        """Search query and returns page annotation result"""

    @abstractmethod
    def _ensure_model(self):
        """Ensure that model or pipeline is loaded"""

    @abstractmethod
    def __len__(self) -> int:
        "Returns the number of documents in current index"

    def index(
        self, documents: list[InputDocument], reinit: bool = True, save: bool = True
    ) -> BasePageRetriever:
        self._ensure_model()
        if reinit:
            self.reinit_indexes()

        for doc in tqdm(documents, desc="Documents"):
            self._index_one(document=doc)

        if save:
            self.save_index()
        logger.warning(f"✓ Indexed {len(self)} pages from {len(documents)} documents")
        return self

    @abstractmethod
    def _index_one(self, document: InputDocument):
        """Index single document"""

    def show(self, results: list[PageAnnotationResult]):
        """Utility to print results."""
        for rank, (doc_version, doc_page, page_annot) in enumerate(results, 1):
            meta_dict = doc_version.document_metadata or {}
            meta = " | ".join(f"{k}={v}" for k, v in meta_dict.items() if k != "format")
            fmt = meta_dict.get("format", "?")
            score = page_annot.score if page_annot.score is not None else 0.0
            norm_score = (
                page_annot.normalized_score
                if page_annot.normalized_score is not None
                else 0.0
            )
            file_path = doc_version.document_path
            page_num = doc_page.page_number

            logger.info(
                f"{rank}. [{score:.3f}] [{norm_score:.3f}] {file_path} ({fmt}) — page {page_num}"  # noqa: E501
                + (f"  ({meta})" if meta else "")
            )

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)


class MultimodalEmbedderRetriever(BasePageRetriever):
    """Multimodal embedder retriever"""

    multi_vector: bool = True

    def __init__(
        self,
        index_path: str | Path = "index.pkl",
        batch_size: int = 4,
        dpi: int = 150,
        extractors: dict[DocumentExtension, PageToImageExtractor] = IMAGE_EXTRACTORS,
    ):
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.dpi = dpi
        self.extractors = extractors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._embeddings: list[np.ndarray] = []
        self._payloads: list[dict] = []
        self._loaded = False

        if self.index_path.exists():
            self.load_index()
        self._ensure_model()

    # ---------- Abstract ----------

    @abstractmethod
    def _load_model(self) -> None: ...

    @abstractmethod
    def _embed_images(self, images: list) -> np.ndarray: ...

    @abstractmethod
    def _embed_query(self, query: str) -> np.ndarray: ...

    # --------- Concrete definitions ---------
    @property
    def page_extractors(self):
        return self.extractors

    # ---------- Persistence ----------

    def save_index(self, index_path: str | Path | None = None):
        path = Path(index_path) if index_path else self.index_path
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self._embeddings, "payloads": self._payloads}, f)
        logger.warning(f"💾 Saved {len(self._payloads)} pages → {path}")

    def load_index(self, index_path: str | Path | None = None) -> BasePageRetriever:
        path = Path(index_path) if index_path else self.index_path
        data = pickle.loads(path.read_bytes())
        self._embeddings = data["embeddings"]
        self._payloads = data["payloads"]
        logger.warning(f"📂 Loaded {len(self._payloads)} pages from {path}")
        return self

    def reinit_indexes(self) -> BasePageRetriever:
        self._embeddings: list[np.ndarray] = []
        self._payloads: list[dict] = []
        logger.warning("Indexes were re-init")
        return self

    def __len__(self) -> int:
        return len(self._payloads)

    # ---------- Lazy model loading ----------

    def _ensure_model(self):
        if not self._loaded:
            logger.warning(f"Loading {self.__class__.__name__} on {self.device}...")
            self._load_model()
            self._loaded = True

    # ---------- Indexing ----------

    def _index_one(self, document: InputDocument):
        doc, doc_version = document.document, document.document_version
        ext = doc.document_ext
        extractor = self.extractors[ext]
        file = Path(doc_version.document_path)
        tqdm.write(f"📄 Extracting {file} ({ext})...")
        try:
            pages = extractor.extract(file, dpi=self.dpi)
        except Exception as e:
            tqdm.write(f"⚠️  Skipping {file.name}: {e}")
            return

        pbar = tqdm(total=len(pages), desc=file.name, leave=False, unit="page")
        for i in range(0, len(pages), self.batch_size):
            chunk = pages[i : i + self.batch_size]
            embs = self._embed_images(chunk)
            for j, emb in enumerate(embs):
                self._embeddings.append(emb)
                self._payloads.append(
                    {
                        "file": file.name,
                        "format": ext,
                        "page": i + j + 1,
                        "document": doc,
                        "document_version": doc_version,
                        **doc_version.document_metadata,
                    }
                )
            pbar.update(len(chunk))
        pbar.close()

    # ---------- Search ----------

    def search(self, query: Query, top_k: int = 10) -> list[PreannotationPageResult]:
        query_text = query.query_text
        if not self._embeddings:
            raise RuntimeError("Index is empty. Call .index() or .load_index() first.")
        self._ensure_model()

        q_emb = self._embed_query(query_text)
        raw_scores = np.array([self._score(q_emb, doc) for doc in self._embeddings])
        normalized = self._minmax(raw_scores)

        mask = np.ones(len(raw_scores), dtype=bool)
        filters = MultiFieldFilterAdapter().to_backend(
            spec=query.get_metadata_filter_spec()
        )

        for key, value in filters.items():
            # Generate a simple boolean list for the current filter
            current_filter_mask = [
                matches_filter(p.get(key), value) for p in self._payloads
            ]
            mask &= np.array(current_filter_mask)
        candidates = np.where(mask, raw_scores, -np.inf)
        normalized = np.where(mask, normalized, 0.0)

        top_idx = np.argsort(-candidates)[:top_k]
        return [
            PreannotationPageResult(
                query=query,
                document=self._payloads[i]["document"],
                document_version=self._payloads[i]["document_version"],
                page=self._payloads[i]["page"],
                score=float(candidates[i]),
                normalized_score=float(normalized[i]),
            )
            for i in top_idx
            if candidates[i] > -np.inf
        ]

    def _score(self, q_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        doc = doc_emb.astype(np.float32)
        if self.multi_vector:
            return (q_emb @ doc.T).max(axis=1).sum()
        return float(q_emb @ doc / (np.linalg.norm(q_emb) * np.linalg.norm(doc) + 1e-9))

    # ---------- Display ----------

    def show_page(self, result: PreannotationPageResult, display_map: bool = False):
        from IPython.display import display

        path = Path(result.document_version.document_path)
        ext = result.document.document_ext
        pages = self.extractors[ext].extract(path, dpi=self.dpi)
        image = pages[result.page - 1]

        if not self.multi_vector or not hasattr(self, "processor") or not display_map:
            display(image)
            return
        self._ensure_model()
        SimilarityMapVisualizer.highlight_image(
            image=image,
            query_text=result.query.query_text,
            model=self.model,
            processor=self.processor,
            device=self.device,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} pages={len(self)}>"


# ============================================================
# Concrete retrievers
# ============================================================


class ColPaliRetriever(MultimodalEmbedderRetriever):
    """ColPali multi-vector late-interaction retriever."""

    multi_vector = True

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        **kwargs,
    ):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self):
        from colpali_engine.models import ColPali, ColPaliProcessor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = ColPali.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

    def _embed_images(self, images):
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query):
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()


class ColQwen2Retriever(MultimodalEmbedderRetriever):
    """ColQwen2 multi-vector late-interaction retriever."""

    multi_vector = True

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        **kwargs,
    ):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = ColQwen2.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)

    def _embed_images(self, images):
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query):
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()


class NomicDenseRetriever(MultimodalEmbedderRetriever):
    """Nomic Embed Multimodal — dense single-vector, faster + smaller storage."""

    multi_vector = False

    def __init__(
        self, model_name: str = "nomic-ai/nomic-embed-multimodal-3b", **kwargs
    ):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self):
        from colpali_engine.models import BiIdefics3, BiIdefics3Processor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = BiIdefics3.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = BiIdefics3Processor.from_pretrained(self.model_name)

    def _embed_images(self, images):
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query):
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()


class TomoroColQwen3Retriever(MultimodalEmbedderRetriever):
    """TomoroAI ColQwen3 multi-vector late-interaction retriever."""

    multi_vector = True

    def __init__(self, model_name: str = "TomoroAI/tomoro-colqwen3-embed-4b", **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = ColQwen2.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)

    def _embed_images(self, images):
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query):
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()


class ColModernVBertRetriever(MultimodalEmbedderRetriever):
    """ModernVBERT multi-vector (ColBERT-style) retriever."""

    multi_vector = True

    def __init__(self, model_name: str = "ModernVBERT/colmodernvbert", **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self):
        from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = ColModernVBert.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = ColModernVBertProcessor.from_pretrained(self.model_name)

    def _embed_images(self, images):
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query):
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()


class ModernVBertRetriever(MultimodalEmbedderRetriever):
    """ModernVBert single-vector (Dense) retriever."""

    multi_vector = False

    def __init__(self, model_name: str = "ModernVBERT/modernvbert", **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self):
        from colpali_engine.models import BiModernVBert, BiModernVBertProcessor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = BiModernVBert.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = BiModernVBertProcessor.from_pretrained(self.model_name)

    def _embed_images(self, images):
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query):
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()
