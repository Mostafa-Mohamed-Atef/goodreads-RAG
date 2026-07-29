"""Unit tests for rag.generator.Generator."""

from unittest.mock import MagicMock, patch


def test_answer_calls_groq_and_strips():
    """Generator should call the Groq API and strip whitespace from the answer."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "  Dune is a classic sci-fi novel.  "

    with patch("rag.generator.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock_resp

        from rag.generator import Generator

        gen = Generator()
        result = gen.answer(
            "What is Dune?",
            ["Dune is a sci-fi novel by Frank Herbert."],
        )

    assert result == "Dune is a classic sci-fi novel."
    MockGroq.return_value.chat.completions.create.assert_called_once()


def test_answer_includes_context_in_prompt():
    """The user prompt sent to Groq should contain the context chunks."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Answer."

    with patch("rag.generator.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock_resp

        from rag.generator import Generator

        gen = Generator()
        gen.answer("Test?", ["Chunk A", "Chunk B"])

    call_args = MockGroq.return_value.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_content = messages[1]["content"]

    assert "Chunk A" in user_content
    assert "Chunk B" in user_content
    assert "Test?" in user_content
