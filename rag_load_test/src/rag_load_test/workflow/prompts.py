"""Chat-message construction for the ``generate`` stage.

Example::

    messages = build_rag_messages("How many vacation days?", reranked_passages)
    answer = await chat_model.generate(messages)
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts.models import ScoredPassage

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite passages as [n]. If the context is insufficient, say so."
)


def format_context(passages: Sequence[ScoredPassage]) -> str:
    """Render passages as ``[1] text`` lines; numbering matches the citations asked for.

    Example::

        format_context([p_a, p_b])  # "[1] ...\\n[2] ..."
    """
    return "\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, start=1))


def build_rag_messages(
    question: str, passages: Sequence[ScoredPassage]
) -> list[dict[str, str]]:
    """Build the OpenAI-style message list: system prompt + numbered context + question.

    Example::

        build_rag_messages("q?", [p])
        # [{"role": "system", ...},
        #  {"role": "user", "content": "Context:\\n[1] ...\\n\\nQuestion: q?"}]
    """
    user_content = f"Context:\n{format_context(passages)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
