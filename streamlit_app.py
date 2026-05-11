"""
Streamlit UI for the Goodreads RAG pipeline.
Run from project root: streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from RAG.pipeline import DEFAULT_TOP_K, RAGPipeline


@st.cache_resource(show_spinner="Loading embeddings and FAISS index…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(verbose=False)


def main() -> None:
    st.set_page_config(
        page_title="Goodreads RAG",
        page_icon="📚",
        layout="wide",
    )

    st.title("Goodreads top books — RAG Q&A")
    st.caption(
        "Ask questions about the scraped Goodreads corpus. "
        "Answers use retrieved book fields only (Groq + local embeddings)."
    )

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.info("Set `GROQ_API_KEY` in a `.env` file or your environment, then refresh.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider(
            "Books to retrieve (top-k)",
            min_value=1,
            max_value=20,
            value=min(5, DEFAULT_TOP_K),
            help="More context can help broad questions; fewer keeps focus.",
        )
        show_sources = st.checkbox("Show retrieved snippets", value=True)
        st.divider()
        st.markdown(
            "**Stack:** SentenceTransformers → FAISS → Groq (`llama-3.1-8b-instant`)"
        )

    question = st.text_area(
        "Your question",
        placeholder="e.g. Which highly rated books are in Arabic?",
        height=100,
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        ask = st.button("Ask", type="primary", use_container_width=True)

    if ask and question.strip():
        with st.spinner("Retrieving relevant books…"):
            hits = pipeline.retrieve_with_meta(question.strip(), top_k=top_k)
            chunks = [h[0]["text"] for h in hits]

        with st.spinner("Generating answer…"):
            answer = pipeline.generate_answer(question.strip(), chunks)

        st.subheader("Answer")
        st.markdown(answer)

        if show_sources:
            st.subheader("Sources (retrieved rows)")
            for i, (row, dist) in enumerate(hits, start=1):
                title = row.get("title") or "(no title)"
                author = row.get("author") or ""
                snippet = (row.get("text") or "")[:600]
                with st.expander(f"{i}. {title} — {author} (distance: {dist:.4f})"):
                    st.text(snippet + ("…" if len(row.get("text") or "") > 600 else ""))
    elif ask:
        st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
