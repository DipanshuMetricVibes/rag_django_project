import os
import re
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 📁 Paths
CHUNK_DIR = "chunks"
INDEX_DIR = "index"
CHUNK_META_DIR = "chunk-metadata"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHUNK_META_DIR, exist_ok=True)

# 🧠 Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 📦 Step 1: Load and clean chunks
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = re.split(r'--- Chunk \d+ ---\n', content)
    return [c.strip() for c in chunks if c.strip()]

# 🧠 Step 2: Generate Embeddings + FAISS index
def create_index_for_file(file_name):
    base = file_name.replace("-chunks.txt", "")
    chunk_path = os.path.join(CHUNK_DIR, file_name)
    chunks = load_chunks(chunk_path)

    print(f"📄 Embedding {len(chunks)} chunks from {file_name}...")

    embeddings = model.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Save index
    index_file = os.path.join(INDEX_DIR, f"{base}-index.index")
    faiss.write_index(index, index_file)

    # Save metadata
    meta_file = os.path.join(CHUNK_META_DIR, f"{base}-chunks.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved index ➤ {index_file}")
    print(f"✅ Saved metadata ➤ {meta_file}")

def process_all_chunk_files():
    for file in os.listdir(CHUNK_DIR):
        if file.endswith("-chunks.txt"):
            create_index_for_file(file)
    print("\n🚀 All embeddings created and stored in FAISS.")

if __name__ == "__main__":
    process_all_chunk_files()
