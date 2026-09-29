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

print("Total pages extracted:", len(documents))

for doc in documents:
    print("\n--------------------")
    print("Source:", doc["source"])
    print("Page:", doc["page"])
    print("Text:", doc["text"][:200])