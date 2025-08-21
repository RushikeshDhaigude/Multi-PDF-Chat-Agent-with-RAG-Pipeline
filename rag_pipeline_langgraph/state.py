# state.py
from typing import TypedDict, List
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

class PDF_CHAT_STATE(TypedDict):
    question: str # The question to be answered
    pdf_paths: List[str]  # List of paths to the PDF files
    chunks: List[Document] # List of document chunks created from the PDFs
    vectordb: Chroma # Vector store for the PDF chunks
    retrieved_docs: List[Document] 
