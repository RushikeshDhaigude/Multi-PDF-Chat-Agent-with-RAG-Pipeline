# rag_pipeline/utils.py
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# cleaning utilities
def clean_text(text: str) -> str:
    # basic cleaning: remove repeated whitespace and common PDF artefacts
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def make_text_chunks(texts, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    return splitter.split_documents(texts)