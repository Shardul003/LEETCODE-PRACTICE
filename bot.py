import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# Load or create collection
collection = chroma_client.get_or_create_collection(name="rag_chunks")

# Load all chunks and metadata for fallback keyword search
def load_all_chunks_with_metadata(chunk_folders):
    texts = []
    metadata = []
    for folder in chunk_folders:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts.append(content)
                    metadata.append({"source": path})
    return texts, metadata

chunk_folders = [r"C:\RAG\Chunks", r"C:\RAG\Chunks1"]
all_texts, all_metadata = load_all_chunks_with_metadata(chunk_folders)

# Semantic search
def search_chunks(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

# Keyword fallback search
def keyword_fallback(query, top_k=5):
    query_lower = query.lower()
    matches = []
    for text, meta in zip(all_texts, all_metadata):
        if query_lower in text.lower():
            matches.append((text, meta))
            if len(matches) >= top_k:
                break
    return matches

# Format response
def format_response(results, fallback_matches=None):
    if results['documents'] and results['documents'][0]:
        response = ""
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            distance = results['distances'][0][i]
            response += f"\n📄 Result {i+1} (from {os.path.basename(source)} — score: {distance:.4f}):\n{doc}\n{'-'*60}"
        return response
    elif fallback_matches:
        response = "🔍 No semantic match found, but here are keyword matches:\n"
        for i, (text, meta) in enumerate(fallback_matches):
            response += f"\n📄 Fallback {i+1} (from {os.path.basename(meta['source'])}):\n{text}\n{'-'*60}"
        return response
    else:
        return "😕 No relevant results found. Try rephrasing your question."

# Main loop
if __name__ == "__main__":
    print("🤖 RAG Bot is ready! Type your question below.\n")
    while True:
        user_query = input("🗨️ You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        if not user_query:
            continue
        results = search_chunks(user_query)
        fallback_matches = keyword_fallback(user_query)
        answer = format_response(results, fallback_matches)
        print(answer)