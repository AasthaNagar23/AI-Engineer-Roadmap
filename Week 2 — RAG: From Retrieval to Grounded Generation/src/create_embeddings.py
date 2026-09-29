from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os
import faiss
import json
import re

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Read all PDFs
documents_folder = "Documents"
documents = []

for filename in os.listdir(documents_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(documents_folder, filename)
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():

                documents.append({
                    "text": text,
                    "source": filename,
                    "page": page_number
                })

def clean_text(text):
    # Multiple spaces, tabs aur newlines ko single space mein convert karega
    text = re.sub(r"\s+", " ", text)

    # Starting aur ending ke extra spaces remove karega
    text = text.strip()

    return text

#create_chunks
def create_chunks(text, chunk_size=500, overlap=100):
    # Sabse pehle text clean karenge
    text = clean_text(text)

    chunks = []

    start = 0

    # Har next chunk overlap ke baad start hoga
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size

        # Text ka ek chunk create karo
        chunk = text[start:end]

        # Empty chunk save nahi karna
        if chunk.strip():
            chunks.append(chunk.strip())

        # Next chunk ke liye move karo
        start += step

    return chunks

# 4. Create chunks with metadata
all_chunks = []

for doc in documents:

    chunks = create_chunks(doc["text"])

    for chunk in chunks:

        all_chunks.append({
            "text": chunk,
            "source": doc["source"],
            "page": doc["page"]
        })


# 5. Convert chunks into embeddings
texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(texts)

print("Total chunks:", len(all_chunks))
print("Embedding shape:", embeddings.shape)
print("First chunk:")
print(all_chunks[0]["text"][:300])

# -----------------------------
# 6. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print("Total vectors in FAISS:", index.ntotal)
print("Vector dimension:", index.d)


# -----------------------------
# 7. Save FAISS index
# -----------------------------

faiss.write_index(index, "vector.index")


# -----------------------------
# 8. Save chunks + metadata
# -----------------------------

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, ensure_ascii=False, indent=2)

print("Chunks saved as chunks.json")

