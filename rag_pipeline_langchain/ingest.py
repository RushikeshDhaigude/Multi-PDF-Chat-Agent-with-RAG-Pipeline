# rag_pipeline/ingest.py
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from rag_pipeline_langchain.utils import clean_text


def load_pdf_to_docs(path: str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()  # returns list[Document]
    cleaned = []
    for p in pages:
        text = clean_text(p.page_content)
        # preserve metadata for citation
        cleaned.append(Document(page_content=text, metadata={**(p.metadata or {}), "source": path}))
    return cleaned


# helper to load multiple PDFs

def load_multiple_pdfs(paths: list[str]):
    all_docs = []
    for p in paths:
        all_docs.extend(load_pdf_to_docs(p))
    return all_docs