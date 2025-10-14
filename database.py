import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your chunk folders
chunk_folders = [r"C:\RAG\Chunks", r"C:\RAG\Chunks1"]

# Load all text chunks from both folders
def load_all_chunks(folders):
    texts = []
    metadata = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts.append(content)
                    metadata.append({"source": path})
    return texts, metadata

# Embed the chunks
def embed_chunks(texts):
    return model.encode(texts, convert_to_numpy=True)

# Save FAISS index and metadata
def save_faiss_index(index, metadata, index_path="faiss_index.index", meta_path="metadata.pkl"):
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

# Main pipeline
texts, metadata = load_all_chunks(chunk_folders)
embeddings = embed_chunks(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata
save_faiss_index(index, metadata)

print(f"✅ Stored {len(texts)} chunks in FAISS index.")