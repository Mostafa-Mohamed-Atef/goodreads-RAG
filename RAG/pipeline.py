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
openai.api_key = os.getenv("GROQ_API_KEY")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Generative model
GEN_MODEL = "gpt-3.5-turbo"

# Number of retrieved chunks
TOP_K = 5

# -----------------------------
# 2. LOAD YOUR SCRAPED DATA
# -----------------------------
# Example: CSV with columns: id, text, title, author, source_url
data = pd.read_csv("scraped_books.csv")
texts = data["text"].tolist()

# -----------------------------
# 3. EMBEDDINGS
# -----------------------------
print("Generating embeddings...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
embeddings = embedder.encode(texts, convert_to_numpy=True)

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
def retrieve_chunks(question, top_k=TOP_K):
    question_vec = embedder.encode([question])
    distances, indices = index.search(question_vec, top_k)
    retrieved = [metadata[i]["text"] for i in indices[0]]
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
    
    response = openai.ChatCompletion.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    return response['choices'][0]['message']['content'].strip()

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