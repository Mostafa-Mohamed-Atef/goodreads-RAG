# RAG Pipeline: Python Implementation
# Requirements: pip install openai faiss-cpu pandas sentence-transformers

import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
# Set your OpenAI API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Generative model
GEN_MODEL = "llama-3.1-8b-instant"

# Number of retrieved chunks
TOP_K = 10

# -----------------------------
# 2. LOAD YOUR SCRAPED DATA
# -----------------------------
# Example: CSV with columns: id, text, title, author, source_url
data = pd.read_csv("Data/goodreads_top100_full.csv")
cols_to_embed = ["title", "author", "rating", "ratings_count", "reviews_count", "description", "format", "language", "published"]
data["text"] = data.apply(lambda row: "\n".join([f"{col.capitalize()}: {row[col]}" for col in cols_to_embed if pd.notna(row[col])]), axis=1)
texts = data["text"].tolist()

# -----------------------------
# 3. EMBEDDINGS
# -----------------------------
print("Generating embeddings...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
embeddings = embedder.encode(texts, convert_to_numpy=True)
print("Total rows in CSV:", len(data))
print("Total texts:", len(texts))
print("Embeddings shape:", embeddings.shape)
# -----------------------------
# 4. VECTOR STORE (FAISS)
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# Optional: store mapping from index -> metadata
metadata = data.to_dict(orient="records")

# -----------------------------
# 5. RETRIEVAL FUNCTION
# -----------------------------
def retrieve_chunks(question, top_k=5):
    question_vec = embedder.encode([question])
    distances, indices = index.search(question_vec, top_k)

    print("\nTop indices:", indices[0])
    print("\nDistances:", distances[0])

    retrieved = []
    for i in indices[0]:
        print("\n--- Chunk Preview ---")
        print(metadata[i]["text"][:200])
        retrieved.append(metadata[i]["text"])

    return retrieved

# -----------------------------
# 6. GENERATION FUNCTION
# -----------------------------
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    Answer the question using ONLY the context below. 
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {question}
    """
    
    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

# -----------------------------
# 7. EXAMPLE USAGE
# -----------------------------
while True:
    question = input("Ask a question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    
    chunks = retrieve_chunks(question)
    answer = generate_answer(question, chunks)
    print("\nAnswer:", answer)
    print("-" * 50)