# rag_pipeline/embeddings_store.py
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
import shutil


# Load environment variables from the .env file
load_dotenv()

# # Access the variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

def create_or_load_vectorstore(docs, persist_directory="./chroma_new_db", embedding_model=None):
    """
    docs: list[Document]
    embedding_model: instance of GoogleGenerativeAIEmbeddings (or compatible)
    """

    # Initialize embedding model if not provided
    if embedding_model is None:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",  # Gemini embeddings compatible
            google_api_key=gemini_api_key,
            client=None,  # None forces sync calls in some versions
        )

    # If persist directory exists, load DB; else create new
    if os.path.exists(persist_directory):
        print("Loading existing vector DB")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    else:
        print("Creating new vector DB")
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vectordb.persist()

    return vectordb


# Helper to initialize embedder
def make_embedder():
    # LangChain/google-genai integration will read env vars like GOOGLE_API_KEY
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=gemini_api_key,
        client=None,  # None forces sync calls in some versions
    )
