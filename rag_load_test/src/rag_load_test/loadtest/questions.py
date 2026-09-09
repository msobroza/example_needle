"""Question bank for the load test, derived from the synthetic corpus templates.

Using the same templates (and default seed) as ``rag-ingest --synthetic`` keeps
retrieval hits meaningful instead of returning random passages.

Example::

    from rag_load_test.loadtest.questions import QUESTIONS
    question = pick_question(random.Random(), QUESTIONS)
"""

from __future__ import annotations

from ..ingest.corpus import synthetic_questions

QUESTION_COUNT = 50

QUESTIONS: list[str] = synthetic_questions(QUESTION_COUNT)
