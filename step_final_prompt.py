import os
import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from google import genai

# ------------------ CONFIG ------------------
PROJECT_ID = "metricvibes-1718777660306"
REGION = "us-central1"

CHUNK_METADATA_DIR = "chunks"         # folder containing: report1-chunks.txt
FAISS_INDEX_DIR = "index"             # folder containing: report1-index.index
USER_MAPPING_FILE = "user_mapping.json"

# ------------------ INIT ------------------

# Init Gemini
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

# Init Embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------ LOAD USER MAPPING ------------------

with open(USER_MAPPING_FILE, "r") as f:
    user_mapping = json.load(f)

# ------------------ Function ------------------


def ask_gemini_rag(user_name, query):
    allowed_reports = user_mapping.get(user_name)
    
    if not allowed_reports:
        print(f"❌ User '{user_name}' not found or not authorized.")
        return

    print(f"\n✅ {user_name} can access: {allowed_reports}")
    query_vector = embedding_model.encode([query]).astype("float32")

    all_matches = []

    for report in allowed_reports:
        index_path = os.path.join(FAISS_INDEX_DIR, f"{report}-index.index")
        metadata_path = os.path.join(CHUNK_METADATA_DIR, f"{report}-chunks.txt")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"⚠️ Skipping {report} (missing index or metadata)")
            continue

        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_raw = f.read()

        chunks = [chunk.strip() for chunk in chunks_raw.split("--- Chunk ") if chunk.strip()]
        chunks = [re.split(r'\n', c, maxsplit=1)[1] if '\n' in c else c for c in chunks]

        distances, indices = index.search(query_vector, 5)

        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                all_matches.append((distances[0][i], chunks[idx], report, idx))  # include report and chunk index

    if not all_matches:
        print("⚠️ No matches found in allowed documents.")
        return

    all_matches.sort(key=lambda x: x[0])  # Sort by distance
    top_matches = all_matches[:5]

    print("\n📊 Top Matched Chunks:")
    for i, (distance, chunk, report_name, chunk_index) in enumerate(top_matches, start=1):
        print(f"\n🔹 Match {i}")
        print(f"📄 Report: {report_name}")
        print(f"📌 Chunk Index: {chunk_index}")
        print(f"📏 Distance: {distance:.4f}")
        print(f"🧠 Content Preview:\n{chunk[:400]}...")  # Only preview first 400 characters

    # Final context for Gemini
    context = "\n\n".join([match[1] for match in top_matches])
    prompt = f"""Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        print("\n🤖 Gemini's Response:\n")
        print(response.text)
    except Exception as e:
        print(f"❌ Gemini error: {e}")


# ------------------ MAIN ------------------

if __name__ == "__main__":
    user_name = input("👤 Enter your name: ").strip()

    while True:
        query = input("\n🧠 Ask something (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        ask_gemini_rag(user_name, query)
