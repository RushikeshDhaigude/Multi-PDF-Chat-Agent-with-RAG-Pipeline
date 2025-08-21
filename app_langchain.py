# app.py
import streamlit as st
from rag_pipeline_langchain.ingest import load_multiple_pdfs
from rag_pipeline_langchain.utils import make_text_chunks
from rag_pipeline_langchain.embeddings_store import create_or_load_vectorstore, make_embedder
from rag_pipeline_langchain.qa_agent import make_qa_chain, run_qa
import tempfile
import asyncio
import os


# Ensure event loop exists (fix Streamlit async issue)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit Page Config
st.set_page_config(page_title="PDF Chat - RAG Demo", layout="wide")
st.title("PDF Chat Agent (Langchain + Gemini)")

# File uploader
uploaded_files = st.file_uploader("Upload PDF(s)",type=["pdf"],accept_multiple_files=True)


# Session state for vector DB
if 'vectordb' not in st.session_state:
    st.session_state['vectordb'] = None


# Ingest PDFs, create embeddings
if uploaded_files:
    with st.spinner("Ingesting PDFs..."):
        paths = []
        for f in uploaded_files:
            # Save PDFs to temporary files
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tf.write(f.read())
            tf.flush()
            paths.append(tf.name)

        # Load PDFs and create document objects
        docs = load_multiple_pdfs(paths)

        # Chunk documents
        chunks = make_text_chunks(docs)

        # Initialize embeddings
        embeddor = make_embedder()

        # Create or load vector DB
        vectordb = create_or_load_vectorstore(
            chunks,
            persist_directory="./chroma_new_db",
            embedding_model=embeddor
        )
        st.session_state['vectordb'] = vectordb
        st.success("PDFs ingested and embedded successfully.")


# Question answering interface
if st.session_state.get('vectordb'):
    qa_chain = make_qa_chain(st.session_state['vectordb'])
    query = st.text_input("Ask a question about the uploaded PDFs")
    if query:
        with st.spinner("Searching and generating answer..."):
            resp = run_qa(qa_chain, query)
            st.subheader("Answer")
            print(resp['result'])
            st.write(resp['result'])

            st.subheader("Source documents / snippets")
            for d in resp.get('source_documents', []):
                st.write(d.metadata.get('source'), '-', d.page_content[:400])

