"""ChatModelPort on the OpenAI Chat Completions API (``AsyncOpenAI``)."""

from __future__ import annotations

from typing import Any

from ..errors import RagDependencyError
from ..settings import RagSettings

LLM_COMPONENT = "llm"


class OpenAIChatModel:
    """Async OpenAI chat client with the SDK's timeout and bounded retries.

    ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` are read by the SDK itself, so a
    Domino AI Gateway can be substituted without code changes.

    Example::

        chat = OpenAIChatModel.from_settings(settings)
        answer = await chat.generate([{"role": "user", "content": "hi"}])
    """

    def __init__(self, client: Any, model: str, *, temperature: float = 0.0) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    @classmethod
    def from_settings(cls, settings: RagSettings) -> OpenAIChatModel:
        """Build the real ``AsyncOpenAI`` client from ``RAG_OPENAI_*`` settings."""
        # Imported here so ``import rag_load_test`` stays cheap for the OVMS apps.
        import openai

        client = openai.AsyncOpenAI(
            timeout=settings.openai_timeout_s, max_retries=settings.openai_max_retries
        )
        return cls(client, settings.openai_model)

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=self._temperature
            )
        except Exception as exc:
            translated = _translate_openai_error(exc, self._model)
            if translated is None:
                raise
            raise translated from exc
        return _first_content(response)


def _first_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return choices[0].message.content or ""


def _translate_openai_error(exc: Exception, model: str) -> RagDependencyError | None:
    """Map SDK errors to RagDependencyError; ``None`` for anything else."""
    import openai

    # APITimeoutError subclasses APIConnectionError, so it must be matched first.
    if isinstance(exc, openai.APITimeoutError):
        return RagDependencyError(LLM_COMPONENT, str(exc), target=model, kind="timeout")
    if isinstance(exc, openai.APIStatusError):
        detail = f"HTTP {exc.status_code}: {exc}"
        return RagDependencyError(LLM_COMPONENT, detail, target=model)
    if isinstance(exc, openai.APIError):
        return RagDependencyError(LLM_COMPONENT, str(exc), target=model)
    return None
