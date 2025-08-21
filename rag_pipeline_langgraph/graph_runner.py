from langgraph.graph import StateGraph,START,END
from rag_pipeline_langgraph.state import PDF_CHAT_STATE
from rag_pipeline_langgraph.nodes import ingest_pdfs_node, init_vectordb_node, qa_node

# Initialize empty graph
graph = StateGraph(PDF_CHAT_STATE)

# Register nodes
graph.add_node("ingest", ingest_pdfs_node)
graph.add_node("vector_db", init_vectordb_node)
graph.add_node("qa", qa_node)

# Connect edges (linear flow)
graph.add_edge(START, "ingest")
graph.add_edge("ingest", "vector_db")
graph.add_edge("vector_db", "qa")
graph.add_edge('qa', END)

workflow = graph.compile()


print("Graph initialized with nodes:")