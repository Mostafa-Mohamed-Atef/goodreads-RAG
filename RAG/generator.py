"""LLM answer generation via Groq."""

from __future__ import annotations

import logging
import time

from groq import Groq

from rag.config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert on the scraped Goodreads book corpus. "
    "Answer ONLY based on the provided context. "
    "If the context does not contain enough information, say so."
)


class Generator:
    """Generate a natural-language answer grounded in retrieved context."""

    def __init__(self) -> None:
        log.info("Initialising Groq client — model=%s", settings.gen_model)
        self.client = Groq(api_key=settings.groq_api_key)

    def answer(self, question: str, chunks: list[str]) -> str:
        """Call the Groq LLM with the question + retrieved context chunks."""
        context = "\n\n".join(chunks)
        user_msg = (
            f"Use ONLY the context below to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        log.debug(
            "Generating answer — model=%s, prompt_len=%d chars, %d chunk(s)",
            settings.gen_model,
            len(user_msg),
            len(chunks),
        )
        t0 = time.perf_counter()

        resp = self.client.chat.completions.create(
            model=settings.gen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=settings.temperature,
        )

        elapsed = time.perf_counter() - t0
        answer = resp.choices[0].message.content.strip()

        log.debug("Answer generated in %.2fs — %d chars", elapsed, len(answer))
        return answer
