# nodes.py
import tempfile
from rag_pipeline_langchain.ingest import load_multiple_pdfs
from rag_pipeline_langchain.utils import make_text_chunks
from rag_pipeline_langchain.embeddings_store import create_or_load_vectorstore, make_embedder
from rag_pipeline_langchain.qa_agent import make_qa_chain, run_qa
from rag_pipeline_langgraph.state import PDF_CHAT_STATE


# Node: PDF ingestion + chunking

def ingest_pdfs_node(state :PDF_CHAT_STATE) -> PDF_CHAT_STATE:

    #print(f"Inputs to ingest_pdfs_node: {inputs}")

    pdf_files = state['pdf_paths']

    print(f"Received PDF files: {pdf_files}")

    paths = []
    for f in pdf_files:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf.write(f.read())
        tf.flush()
        paths.append(tf.name)

    # Load PDFs and create document objects

    docs = load_multiple_pdfs(paths)
    chunks = make_text_chunks(docs)

    state['chunks'] = chunks

    #print(state['chunks'])

    return state

# Node: Vector DB creation / load

def init_vectordb_node(state:PDF_CHAT_STATE) -> PDF_CHAT_STATE:
    persist_directory = "./chroma_2_db"

    chunks = state['chunks']
    if not chunks:
        raise ValueError("No chunks found. Run ingest_pdfs_node first.")

    if 'vectordb' not in state or state['vectordb'] is None:
        embeddor = make_embedder()
        vectordb = create_or_load_vectorstore(
            docs=chunks,
            persist_directory=persist_directory,
            embedding_model=embeddor
        )
        state['vectordb'] = vectordb


    return state

# Node: QA with verification
def qa_node(state:PDF_CHAT_STATE) -> PDF_CHAT_STATE:

    vectordb = state["vectordb"]
    if not vectordb:
        raise ValueError("No vector DB found. Run vector_db_node first.")

    question = state.get("question", "")
    # if not question:
    #     raise ValueError("No question provided.")

    # Create QA chain
    qa_chain = make_qa_chain(state['vectordb'], top_k=5)
    resp = run_qa(qa_chain, question)

    state['llm_response'] = resp['result']
    state['retrieved_docs'] = resp.get('source_documents', [])

    return state


print("Nodes initialized:")