from pypdf import PdfReader
import os

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


def create_chunks(text, chunk_size=500, overlap=100):
    
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


all_chunks = []

for doc in documents:

    chunks = create_chunks(doc["text"])

    for chunk in chunks:

        all_chunks.append({
            "text": chunk,
            "source": doc["source"],
            "page": doc["page"]
        })


print("Total chunks:", len(all_chunks))

for i, chunk in enumerate(all_chunks[:5]):

    print("\n--------------------")
    print("Chunk ID:", i)
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:300])