import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.bin")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
sources = metadata["sources"]

# Search function
def search_faiss(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "text": texts[idx],
            "source": sources[idx],
            "score": distances[0][i]
        }
        results.append(result)
    return results

# Format response with summary-style output
def format_response(results):
    if not results:
        return "😕 No relevant results found. Try rephrasing your question."

    summary = "🔎 Summary of key points:\n"
    for res in results:
        lines = res["text"].strip().splitlines()
        first_line = next((line for line in lines if line.strip()), "")
        summary += f"- {first_line.strip()}\n"

    summary += "\n📚 Source files:\n"
    for res in results:
        summary += f"- {os.path.basename(res['source'])} (score: {res['score']:.4f})\n"

    return summary

# Main loop
if __name__ == "__main__":
    print("🤖 FAISS Bot is ready! Type your question below.\n")
    while True:
        user_query = input("🗨️ You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        if not user_query:
            continue
        results = search_faiss(user_query)
        answer = format_response(results)
        print(answer)