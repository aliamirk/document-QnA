import pdfplumber
from pathlib import Path


def load_documents(pdf_path: str):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "page": i + 1,
                    "text": text
                })

    return pages


def chunk_text(text: str, size=250, overlap=50):
    words = text.split()
    chunks = []

    step = size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i + size]
        if len(chunk) < 20:  # skip very small fragments
            continue
        chunks.append(" ".join(chunk))

    return chunks


def chunk_pages(pages, size=250, overlap=50):
    all_chunks = []

    for page in pages:
        chunks = chunk_text(page["text"], size, overlap)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "page": page["page"],
                "chunk_id": idx,
                "text": chunk
            })

    return all_chunks


def generateChunks():
    pdf_path = "/Users/apple/Desktop/document-QA/documents/Finance_and_Stock_Market_FAQs.pdf"

    pages = load_documents(pdf_path)
    chunks = chunk_pages(pages)
    return chunks

