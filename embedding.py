import os
from sentence_transformers import SentenceTransformer

# Choose your embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and accurate

# Define your chunk folders
chunk_folders = [r"C:\RAG\Chunks", r"C:\RAG\Chunks1"]

# Load all text chunks from both folders
def load_all_chunks(folders):
    chunks = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    chunks.append(f.read())
    return chunks

# Embed the chunks
def embed_chunks(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Run the pipeline
chunks = load_all_chunks(chunk_folders)
embeddings = embed_chunks(chunks)

# Print summary
print(f"✅ Embedded {len(chunks)} chunks.")
print(f"🔢 Embedding shape: {embeddings.shape}")