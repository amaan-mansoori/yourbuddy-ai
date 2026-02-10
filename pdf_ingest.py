from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

pdf_docs = []
pdf_index = None

def ingest_pdf(file_path):
    global pdf_docs, pdf_index

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    pdf_docs.extend(chunks)

    embeddings = model.encode(chunks)

    if pdf_index is None:
        pdf_index = faiss.IndexFlatL2(embeddings.shape[1])

    pdf_index.add(np.array(embeddings))


def search_pdf(query, k=3):
    if pdf_index is None:
        return []

    query_vec = model.encode([query])
    _, indices = pdf_index.search(query_vec, k)
    return [pdf_docs[i] for i in indices[0]]
