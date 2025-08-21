# rag_pipeline/qa_agent.py
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# # Access the variables
gemini_api_key = os.getenv("GEMINI_API_KEY")


PROMPT = """
You are an assistant that MUST ONLY ANSWER using the provided documents.\n
Documents: {context}\n
User question: {question}\n
1) Provide a short answer (max 200 words).\n
2) For every factual claim, include a bracketed citation to the source document (use the 'source' metadata).\n
3) At the end, output a 'VERIFICATION' section where you list each claim and the exact supporting text snippet from the documents.\n
If you cannot find supporting text for a claim, say "NOT FOUND" and do NOT attempt to guess.\n
"""


def make_qa_chain(vectordb, top_k=5, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {"temperature": 0.0, "max_output_tokens": 800}


    gemini_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=gemini_api_key,
                                          model_kwargs=model_kwargs,
                                          client=None,  # None forces sync calls in some versions
                                          )

    prompt = PromptTemplate(template=PROMPT,input_variables=["context", "question"])

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    qa_chain = RetrievalQA.from_chain_type(
        llm= gemini_model,
        chain_type="stuff",  # 'map_reduce' or 'refine' are alternatives
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


def run_qa(qa_chain, question: str):
    res = qa_chain(question)
    # res contains 'result' (string) and 'source_documents'
    return res


print("QA agent is ready to use.")