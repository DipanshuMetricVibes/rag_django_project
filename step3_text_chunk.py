import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 📁 Paths
TEXT_DIR = "text-extract"
CHUNK_DIR = "chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

# 🪓 Chunker Config
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def chunk_text_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = splitter.split_text(raw_text)
    print(f"✂️ {len(chunks)} chunks created for {os.path.basename(input_path)}")

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, chunk in enumerate(chunks, 1):
            out.write(f"--- Chunk {idx} ---\n")
            out.write(chunk)
            out.write("\n\n")

def process_all_files():
    for filename in os.listdir(TEXT_DIR):
        if filename.endswith(".txt"):
            base = filename.replace(".txt", "")
            input_path = os.path.join(TEXT_DIR, filename)
            output_path = os.path.join(CHUNK_DIR, f"{base}-chunks.txt")
            chunk_text_file(input_path, output_path)

    print(f"\n✅ All files chunked successfully into '{CHUNK_DIR}' folder.")

if __name__ == "__main__":
    process_all_files()
