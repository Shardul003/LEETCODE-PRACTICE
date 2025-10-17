import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load chunks from folders
chunk_folders = [r"C:\RAG\Chunks1", r"C:\RAG\Chunks2"]

def load_chunks(folders):
    texts = []
    sources = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts.append(content)
                    sources.append(path)
    return texts, sources

texts, sources = load_chunks(chunk_folders)
print(f"📦 Loaded {len(texts)} chunks.")

# Embed all texts
embeddings = model.encode(texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "faiss_index.bin")
with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump({"texts": texts, "sources": sources}, f)

print("✅ FAISS index and metadata saved.")