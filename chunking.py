from semchunk import chunkerify
from transformers import AutoTokenizer
import os

# Load a HuggingFace tokenizer (used by semchunk to count tokens)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Determine a safe chunk_size using the tokenizer's max length (fall back to 512)
# Tokenizer/model limits
model_max_length = getattr(tokenizer, "model_max_length", None) or 512
# Use a conservative chunk size well under the model max to avoid indexing errors.
# Default to 256 tokens or model_max_length-50 whichever is smaller, but not below 128.
chunk_size = max(128, min(256, model_max_length - 50))
# Use a modest overlap (tokens) between adjacent chunks
overlap = min(128, max(0, chunk_size // 10))

# Initialize semchunk chunker using the tokenizer and computed chunk_size
chunker = chunkerify(tokenizer, chunk_size=chunk_size)

# Load your text
with open(r"C:\RAG\2025_2026.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Perform semantic chunking (use overlap to preserve continuity)
chunks = chunker(text, overlap=overlap)

# Post-process chunks: ensure none exceed the tokenizer/model max length.
def split_chunk_by_tokens(chunk_text: str, max_tokens: int, tokenizer, overlap_tokens: int) -> list:
    """Split a chunk by token ids into smaller pieces with overlap."""
    ids = tokenizer.encode(chunk_text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [chunk_text]

    step = max_tokens - overlap_tokens if max_tokens > overlap_tokens else max_tokens
    parts = []
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        slice_ids = ids[start:end]
        text_part = tokenizer.decode(slice_ids, skip_special_tokens=True)
        parts.append(text_part)
        if end == len(ids):
            break

    return parts

# Guard: if semchunk produced a chunk longer than the model allows, split it further.
final_chunks = []
for c in chunks:
    token_count = len(tokenizer.encode(c, add_special_tokens=False))
    if token_count > model_max_length:
        print(f"Chunk too long ({token_count} tokens) — splitting into subchunks of <= {chunk_size} tokens")
        subparts = split_chunk_by_tokens(c, chunk_size, tokenizer, overlap)
        final_chunks.extend(subparts)
    else:
        final_chunks.append(c)

chunks = final_chunks

# Save chunks to folder
output_folder = r"C:\RAG\Chunks2"
os.makedirs(output_folder, exist_ok=True)

for i, chunk in enumerate(chunks, start=1):
    with open(os.path.join(output_folder, f"chunk{i}.txt"), "w", encoding="utf-8") as f:
        f.write(chunk)
        