"""Corpus generation, chunking, JSONL loading and the ingestion CLI (fakes only)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_load_test.contracts.models import Passage
from rag_load_test.ingest.corpus import (
    SYNTHETIC_TOPICS,
    SourceDocument,
    chunk_document,
    load_jsonl_corpus,
    synthetic_corpus,
    synthetic_questions,
)
from rag_load_test.ingest.ingest_cli import build_parser, ingest_passages, main
from rag_load_test.testing.fakes import FakeEmbedder, InMemoryVectorStore

TOPIC_SLUGS = [topic[0] for topic in SYNTHETIC_TOPICS]
SLOT_NAMES = ("{n}", "{days}", "{amount}", "{hours}", "{pct}")


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records how many document batches it embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return await super().embed_documents(texts)


def _words_doc(n_words: int, **metadata: str | int) -> SourceDocument:
    return SourceDocument(
        id="doc-x",
        title="Words",
        text=" ".join(f"w{i}" for i in range(n_words)),
        metadata=dict(metadata),
    )


# --- synthetic corpus ---------------------------------------------------------


def test_synthetic_corpus_is_deterministic_with_sequential_ids() -> None:
    docs = synthetic_corpus(5)
    assert docs == synthetic_corpus(5)
    assert [d.id for d in docs] == [f"doc-{i:04d}" for i in range(5)]
    assert [d.metadata["seq"] for d in docs] == [0, 1, 2, 3, 4]


def test_synthetic_corpus_differs_with_seed_and_between_documents() -> None:
    assert synthetic_corpus(5, seed=1) != synthetic_corpus(5, seed=2)
    docs = synthetic_corpus(20)
    # doc-0000 and doc-0010 share a topic but the numeric slots differ.
    assert docs[0].metadata["topic"] == docs[10].metadata["topic"]
    assert docs[0].text != docs[10].text


def test_synthetic_corpus_cycles_the_ten_topics_in_order() -> None:
    docs = synthetic_corpus(20)
    assert len(TOPIC_SLUGS) == 10
    assert [d.metadata["topic"] for d in docs[:10]] == TOPIC_SLUGS
    assert [d.metadata["topic"] for d in docs[10:]] == TOPIC_SLUGS


def test_synthetic_corpus_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="-1"):
        synthetic_corpus(-1)


def test_synthetic_topics_have_slots_that_all_get_filled() -> None:
    for slug, title, sentences in SYNTHETIC_TOPICS:
        assert len(sentences) >= 5, slug
        assert "Northwind Analytics" in title
        for sentence in sentences:
            assert any(slot in sentence for slot in SLOT_NAMES), (slug, sentence)
    for doc in synthetic_corpus(len(SYNTHETIC_TOPICS)):
        assert "{" not in doc.text and "}" not in doc.text, doc.id
        assert "{" not in doc.title, doc.id


# --- synthetic questions ------------------------------------------------------


def test_synthetic_questions_default_count_and_determinism() -> None:
    questions = synthetic_questions()
    assert len(questions) == 50
    assert questions == synthetic_questions()
    assert all("Northwind Analytics" in q for q in questions)
    assert all("{" not in q for q in questions)


def test_synthetic_questions_cycle_topics() -> None:
    questions = synthetic_questions(10)
    assert len(set(questions)) == 10
    assert "vacation" in questions[0].lower()
    assert synthetic_questions(3) == questions[:3]
    # question i and i+10 target the same topic (numeric slots may differ).
    later = synthetic_questions(20)[10:]
    for first, second in zip(questions, later, strict=True):
        assert first.split()[:3] == second.split()[:3]


# --- chunking -----------------------------------------------------------------


def test_chunk_document_sliding_windows_overlap_ids_and_metadata() -> None:
    doc = _words_doc(300, topic="t", seq=7)
    chunks = chunk_document(doc, chunk_words=100, overlap_words=20)
    assert [c.id for c in chunks] == ["doc-x:0", "doc-x:1", "doc-x:2", "doc-x:3"]
    assert [len(c.text.split()) for c in chunks] == [100, 100, 100, 60]
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert previous.text.split()[-20:] == following.text.split()[:20]
    assert chunks[2].metadata == {
        "topic": "t",
        "seq": 7,
        "doc_id": "doc-x",
        "title": "Words",
        "chunk": 2,
    }


def test_chunk_document_short_document_is_one_chunk() -> None:
    chunks = chunk_document(_words_doc(10))
    assert len(chunks) == 1
    assert chunks[0].text.split() == [f"w{i}" for i in range(10)]


def test_chunk_document_empty_text_yields_no_passages() -> None:
    doc = SourceDocument(id="e", title="Empty", text="   ")
    assert chunk_document(doc) == []


@pytest.mark.parametrize(("chunk", "overlap"), [(120, 120), (10, 30), (0, 0), (5, -1)])
def test_chunk_document_rejects_bad_window(chunk: int, overlap: int) -> None:
    with pytest.raises(ValueError) as excinfo:
        chunk_document(_words_doc(10), chunk_words=chunk, overlap_words=overlap)
    assert str(chunk) in str(excinfo.value) and str(overlap) in str(excinfo.value)


# --- JSONL loader -------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_jsonl_corpus_reads_documents_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"id": "a", "title": "A", "text": "alpha text", "metadata": {"k": 1}},
            "",
            {"id": "b", "title": "B", "text": "beta text"},
        ],
    )
    docs = load_jsonl_corpus(path)
    assert [d.id for d in docs] == ["a", "b"]
    assert docs[0].metadata == {"k": 1}
    assert docs[1].metadata == {}


def test_load_jsonl_corpus_bad_json_reports_line_number(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "bad.jsonl",
        [{"id": "a", "title": "A", "text": "ok"}, "{not json" + "x" * 100],
    )
    with pytest.raises(ValueError) as excinfo:
        load_jsonl_corpus(path)
    message = str(excinfo.value)
    assert "line 2" in message
    assert "{not json" in message
    assert "x" * 100 not in message  # offending line is truncated to 80 chars


def test_load_jsonl_corpus_missing_keys_reports_line_number(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "missing.jsonl", [{"id": "a", "text": "no title"}])
    with pytest.raises(ValueError) as excinfo:
        load_jsonl_corpus(path)
    assert "line 1" in str(excinfo.value)
    assert "title" in str(excinfo.value)


def test_load_jsonl_corpus_non_object_line_reports_line_number(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(tmp_path / "list.jsonl", [["not", "an", "object"]])
    with pytest.raises(ValueError, match="line 1"):
        load_jsonl_corpus(path)


# --- ingest_passages ----------------------------------------------------------


async def test_ingest_passages_batches_and_returns_count() -> None:
    passages = [Passage(id=f"p{i}", text=f"text {i}") for i in range(5)]
    embedder, store = CountingEmbedder(), InMemoryVectorStore()
    count = await ingest_passages(passages, embedder, store, batch_size=2)
    assert count == 5
    assert await store.count() == 5
    assert embedder.batches == [2, 2, 1]


async def test_ingest_passages_empty_is_noop() -> None:
    embedder, store = CountingEmbedder(), InMemoryVectorStore()
    assert await ingest_passages([], embedder, store) == 0
    assert embedder.batches == []


async def test_ingest_passages_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="0"):
        await ingest_passages([], FakeEmbedder(), InMemoryVectorStore(), batch_size=0)


async def test_fake_retrieval_finds_the_right_topic_for_every_question() -> None:
    docs = synthetic_corpus(30)
    passages = [chunk for doc in docs for chunk in chunk_document(doc)]
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    await ingest_passages(passages, embedder, store)
    for slug, question in zip(TOPIC_SLUGS, synthetic_questions(10), strict=True):
        hits = await store.query(await embedder.embed_query(question), top_k=1)
        assert hits[0].metadata["topic"] == slug, question


# --- CLI ----------------------------------------------------------------------


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert (args.synthetic, args.corpus, args.seed) == (200, None, 1234)
    assert (args.chunk_words, args.overlap_words, args.batch_size) == (120, 20, 64)
    assert args.fake is False


def test_build_parser_synthetic_and_corpus_are_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--synthetic", "3", "--corpus", str(tmp_path)])


def test_main_synthetic_fake_prints_summary_line(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no stray .env
    assert main(["--synthetic", "3", "--fake"]) == 0
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["event"] == "ingest_done"
    assert summary["documents"] == 3
    assert summary["passages"] == 3
    assert summary["embedder"] == "fake-embedder"
    assert summary["collection"] == "in-memory"


def test_main_corpus_fake_uses_jsonl_file(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write_jsonl(
        tmp_path / "c.jsonl",
        [
            {"id": "a", "title": "A", "text": " ".join(["w"] * 150)},
            {"id": "b", "title": "B", "text": "short"},
        ],
    )
    argv = ["--corpus", str(path), "--fake", "--chunk-words", "100"]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out.strip())
    assert (summary["documents"], summary["passages"]) == (2, 3)
