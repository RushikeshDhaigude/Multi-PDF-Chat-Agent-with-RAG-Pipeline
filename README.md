# Multi-PDF-Chat-Agent-with-RAG-Pipeline

A Retrieval-Augmented Generation (RAG) chatbot for querying multiple PDF documents with contextual answers, built using LangChain, Google Gemini, and Streamlit.

### Overview

This project implements an end-to-end RAG pipeline to:

1. Accept multiple PDF uploads

2. Extract and preprocess text content

3. Chunk documents for semantic embedding

4. Store embeddings in a vector database (Chroma by default)

5. Use Google Gemini LLM for generating context-aware, source-cited answers

6. Provide an interactive Streamlit UI for querying PDFs


### Features

1. Multi-PDF support : Upload multiple PDFs and query across all documents.

2. Semantic search with embeddings : Uses Gemini embeddings stored in Chroma for fast, accurate retrieval.

3. Strict verification : LLM responses include source citations and fallback if information is missing.

4. Streamlit UI : Intuitive interface for uploading documents, asking questions, and viewing answers with supporting snippets.

### Project Struture 

```
Multi-PDF-RAG-Chat/
│
├─ app.py                  # Streamlit front-end
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation
├─ rag_pipeline_langgraph/  # LangGraph workflow files
│   ├─ state.py            # TypedDict for PDF_CHAT_STATE
│   ├─ nodes.py            # LangGraph nodes (ingest, vector DB, QA)
│   └─ graph_runner.py     # LangGraph workflow runner
├─ rag_pipeline_langchain/  # LangChain workflow files
│   ├─ ingest.py           # PDF ingestion
│   ├─ embeddings_store.py # Embedding creation & persistence
│   ├─ qa_agent.py         # QA chain with Gemini
│   └─ utils.py            # Chunking & cleaning utilities
└─ examples/               # Sample PDFs and outputs
```


### Installation

1. Clone the repository:
```
git clone https://github.com/RushikeshDhaigude/Multi-PDF-Chat-Agent-with-RAG-Pipeline.git
cd Multi-PDF-Chat-Agent-with-RAG-Pipeline
```

2. Create a virtual environment:
```
python -m venv my_env
source my_env/bin/activate   # Linux / macOS
my_env\Scripts\activate      # Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```

5. Set Google Gemini API key in your environment:
```
GEMINI_API_KEY="your_api_key_here"
```

7. Run the Streamlit app :
```
streamlit run app.py
```




