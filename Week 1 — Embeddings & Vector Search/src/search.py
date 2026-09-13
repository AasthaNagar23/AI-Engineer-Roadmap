import faiss
import json

# FAISS index load karo
index = faiss.read_index("vector.index")

# Chunks aur unki information load karo
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("FAISS index loaded")
print("Total vectors:", index.ntotal)
print("Total chunks:", len(chunks))

from sentence_transformers import SentenceTransformer
import numpy as np

# Embedding model load karo
model = SentenceTransformer("all-MiniLM-L6-v2")

# User ki query
query = input("Enter your question: ")

# Query ko vector mein convert karo
query_embedding = model.encode([query])

# FAISS ko float32 format chahiye
query_embedding = np.array(query_embedding).astype("float32")

print("Query embedding shape:", query_embedding.shape)

# Top 3 similar chunks search karo
k = 3

distances, indices = index.search(query_embedding, k)

print("\n===== Search Results =====")

for i in range(k):
    result = chunks[indices[0][i]]

    print(f"\nResult {i + 1}")
    print("Distance:", distances[0][i])
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Text:", result["text"])

