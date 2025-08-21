# app.py
import streamlit as st
from rag_pipeline_langgraph.graph_runner import workflow
from rag_pipeline_langgraph.state import PDF_CHAT_STATE
import asyncio


# Ensure event loop exists (fix Streamlit async issue)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


st.set_page_config(page_title="LangGraph Multi-PDF Chat", layout="wide")
st.title("PDF Chat Agent (LangGraph + Gemini)")

# Initialize state
if 'state' not in st.session_state:
    st.session_state['state'] = PDF_CHAT_STATE(
        question="",
        llm_response="",
        retrieved_docs=[],
        chunks=[],
        vectordb=None,
        pdf_paths=[]
    )

state = st.session_state['state']

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Ingesting PDFs and building vector store..."):
        # Run ingestion step
        state = workflow.invoke(
            {"pdf_paths": uploaded_files},
            config={"configurable": {"step": "ingest"}}
        )

        # Run vector DB step
        state = workflow.invoke(
            {},
            config={"configurable": {"step": "vector_db"}}
        )

        st.success("PDFs ingested and vector DB ready.")

# Ask questions
if state.get('vectordb'):
    query = st.text_input("Ask a question about the uploaded PDFs")
    if query:
        with st.spinner("Generating verified answer..."):
            state = workflow.invoke(
                {"question": query},
                config={"configurable": {"step": "qa"}}
            )

            st.subheader("Answer")
            st.write(state['llm_response'])

            st.subheader("Source documents / snippets")
            for d in state['retrieved_docs']:
                st.write(d.metadata.get('source'), "-", d.page_content[:400])
