
import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# --------------------------------
# 1. Find Project Root Directory
# --------------------------------

# __file__ = current file (rag_pipeline.py)
# dirname(__file__) = src folder
# dirname(dirname(__file__)) = project root

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# --------------------------------
# 2. Define File Paths
# --------------------------------

# FAISS index aur chunks.json
# project ke root folder mein hain

index_path = os.path.join(
    BASE_DIR,
    "vector.index"
)

chunks_path = os.path.join(
    BASE_DIR,
    "chunks.json"
)


# --------------------------------
# 3. Load FAISS Index
# --------------------------------

index = faiss.read_index(index_path)


# --------------------------------
# 4. Load Chunks and Metadata
# --------------------------------

with open(
    chunks_path,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


print("FAISS index loaded")
print("Total vectors:", index.ntotal)
print("Total chunks:", len(chunks))


# --------------------------------
# 5. Load Embedding Model
# --------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 6. Retrieve Relevant Chunks
# --------------------------------

def retrieve_chunks(query, k=3):

    # User query ko vector mein convert karna
    query_embedding = model.encode([query])

    # FAISS ke liye float32 format
    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # Agar index mein 3 se kam chunks hain,
    # toh available chunks tak hi search karenge
    actual_k = min(k, index.ntotal)

    # Similar chunks search karna
    distances, indices = index.search(
        query_embedding,
        actual_k
    )

    retrieved_chunks = []

    for i in range(actual_k):

        # Retrieved chunk ka index
        chunk_index = indices[0][i]

        # Invalid index ko skip karna
        if chunk_index == -1:
            continue

        # Original chunk aur metadata access karna
        result = chunks[chunk_index]

        retrieved_chunks.append({

            "text": result["text"],

            "source": result["source"],

            "page": result["page"],

            "distance": float(
                distances[0][i]
            )

        })

    return retrieved_chunks


# --------------------------------
# 7. Create Context
# --------------------------------

def create_context(retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        # Har chunk ke saath source aur page
        # preserve kar rahe hain

        context_parts.append(

            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Text: {chunk['text']}"

        )

    # Sabhi chunks ko ek context mein combine karna
    context = "\n\n".join(
        context_parts
    )

    return context


# --------------------------------
# 8. Build RAG Prompt
# --------------------------------

def build_prompt(context, query):
    prompt = f"""
You are a strict document-based question-answering assistant.

Your task is to answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not infer or assume any information.
3. Do not mention technologies unless they are explicitly stated in the context.
4. If the answer is not clearly available, respond with exactly:
Information not available in the documents.
5. Give a short and direct answer.
6. Do not provide both an inferred answer and the unavailable message.

Provided Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


def generate_answer(prompt):
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# --------------------------------
# 9. Main Program
# --------------------------------

if __name__ == "__main__":

    # User se question lena
    query = input(
        "\nEnter your question: "
    )

    # Relevant chunks retrieve karna
    retrieved_chunks = retrieve_chunks(
        query,
        k=3
    )

    # Retrieved chunks ko context mein convert karna
    context = create_context(
        retrieved_chunks
    )

    # Context aur query se prompt banana
    prompt = build_prompt(
        context,
        query
    )

    # --------------------------------
    # Print Retrieved Chunks
    # --------------------------------

    print(
        "\n========== RETRIEVED CHUNKS =========="
    )

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        print(
            f"\nResult {i + 1}"
        )

        print(
            "Source:",
            chunk["source"]
        )

        print(
            "Page:",
            chunk["page"]
        )

        print(
            "Distance:",
            chunk["distance"]
        )

        print(
            "Text:",
            chunk["text"][:300]
        )

    # --------------------------------
    # Print Combined Context
    # --------------------------------

    print(
        "\n========== COMBINED CONTEXT =========="
    )

    print(context)

    # --------------------------------
    # Print RAG Prompt
    # --------------------------------

    print(
        "\n========== RAG PROMPT =========="
    )

    print(prompt)

    answer = generate_answer(prompt)

    print("\n========== FINAL ANSWER ==========")
    print(answer)