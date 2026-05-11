# RAG Pipeline: Python Implementation
# Requirements: pip install openai faiss-cpu pandas sentence-transformers groq python-dotenv

from __future__ import annotations

import os
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOP_K = 10


class RAGPipeline:
    """Goodreads book corpus: embed, retrieve with FAISS, answer with Groq."""

    def __init__(self, data_dir: Path | str | None = None, verbose: bool = True):
        self.verbose = verbose
        data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set in environment or .env")

        # Load all goodreads CSV files
        csv_files = list(data_dir.glob("goodreads_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No goodreads CSV files found in {data_dir}")

        if self.verbose:
            print(f"Loading data from: {[f.name for f in csv_files]}")

        data_frames = []
        for f in csv_files:
            df = pd.read_csv(f)
            data_frames.append(df)
        
        data = pd.concat(data_frames, ignore_index=True)
        
        # Deduplicate by URL if possible
        if "url" in data.columns:
            data = data.drop_duplicates(subset=["url"])

        cols_to_embed = [
            "title",
            "author",
            "rating",
            "ratings_count",
            "reviews_count",
            "description",
            "format",
            "language",
            "published",
        ]
        
        # Ensure columns exist before embedding
        available_cols = [c for c in cols_to_embed if c in data.columns]
        
        data["text"] = data.apply(
            lambda row: "\n".join(
                [f"{col.capitalize()}: {row[col]}" for col in available_cols if pd.notna(row[col])]
            ),
            axis=1,
        )
        self.texts = data["text"].tolist()
        self.metadata = data.to_dict(orient="records")

        if self.verbose:
            print(f"Generating embeddings for {len(self.texts)} items...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = self.embedder.encode(self.texts, convert_to_numpy=True)
        if self.verbose:
            print("Embeddings shape:", embeddings.shape)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        if self.verbose:
            print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve_chunks(self, question: str, top_k: int = 5) -> list[str]:
        question_vec = self.embedder.encode([question])
        distances, indices = self.index.search(question_vec, top_k)

        if self.verbose:
            print("\nTop indices:", indices[0])
            print("\nDistances:", distances[0])

        retrieved: list[str] = []
        for i in indices[0]:
            if self.verbose:
                print("\n--- Chunk Preview ---")
                print(self.metadata[i]["text"][:200])
            retrieved.append(self.metadata[i]["text"])
        return retrieved

    def retrieve_with_meta(
        self, question: str, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """Return (row dict, L2 distance) for each hit."""
        question_vec = self.embedder.encode([question])
        distances, indices = self.index.search(question_vec, top_k)
        out: list[tuple[dict, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            out.append((self.metadata[idx], float(dist)))
        return out

    def generate_answer(self, question: str, retrieved_chunks: list[str]) -> str:
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
    You are an expert on the scraped Goodreads book corpus.
    Use the provided context to answer the user's question accurately.

    - Answer ONLY based on the context below.
    - If the context does not contain enough information to answer the question, or if the question is unrelated to the books provided, say "I'm sorry, but I don't have enough information in my database to answer that."
    - Do NOT start your answer with "I don't know" if you are going to provide information from the context.

    Context:
    {context}

    Question:
    {question}
    """

        response = self.client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()


def main() -> None:
    pipeline = RAGPipeline(verbose=True)
    while True:
        question = input("Ask a question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        chunks = pipeline.retrieve_chunks(question, top_k=DEFAULT_TOP_K)
        answer = pipeline.generate_answer(question, chunks)
        print("\nAnswer:", answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
