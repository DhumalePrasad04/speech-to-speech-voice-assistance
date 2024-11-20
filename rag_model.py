import os
import getpass
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader, Docx2txtLoader
from langsmith import expect
from pdfplumber import open as pdfplumber_open
import pandas as pd

def set_env(var:str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getuser(f"{var}")
set_env("TAVAILY_API_KEY")
os.environ["TOKENIZER_PARALLELISM"]="true"
local_llm="llama3.2:1b"
llm=ChatOllama(model=local_llm,temperature=0)
llm_json_mode=ChatOllama(model=local_llm,temperature=0)

def load_doc_from_url(urls):
    doc=[]
    for url in urls:
        try:
            loader=WebBaseLoader(url)
            docs=loader.load()
            doc.extend(docs)
        except Exception as e:
            print(e)
    return doc
def load_documents(file_paths):
    """
    Load documents from PDF, DOCX, and TXT files with robust handling for complex data structures like tables.
    """
    documents = []
    for file_path in file_paths:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Determine file type and process accordingly
            if file_path.endswith(".pdf"):
                with pdfplumber_open(file_path) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            df = pd.DataFrame(table)
                            documents.append({"type": "table", "content": df})
                        text = page.extract_text()
                        if text:
                            documents.append({"type": "text", "content": text})

            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                for table in doc.tables:
                    table_data = [
                        [cell.text.strip() for cell in row.cells]
                        for row in table.rows
                    ]
                    df = pd.DataFrame(table_data)
                    documents.append({"type": "table", "content": df})
                paragraphs = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
                if paragraphs:
                    documents.append({"type": "text", "content": paragraphs})

            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    if text:
                        documents.append({"type": "text", "content": text})

            else:
                print(f"Unsupported file format: {file_path}. Skipping...")

        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    if not documents:
        print("No documents were loaded. Please check file paths and formats.")

    return documents
def chunk_documents(documents,chunk_size=1000,chunk_overlap=200):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# Step 4: Create Vectorstore with Ollama Embeddings
def create_chromadb_with_ollama_embeddings(documents, persist_directory="./vectorstore"):
    """Embed documents using Ollama embeddings and store them in ChromaDB."""
    embedding_model = OllamaEmbeddings(model=local_llm)
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

# Step 5: Initialize RAG Pipeline
def initialize_rag_pipeline(vectorstore_path):
    """Initialize a Retrieval-Augmented Generation pipeline."""
    embedding_model = OllamaEmbeddings(model=local_llm)
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=local_llm, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 6: Web Search Tool
web_search_tool = TavilySearchResults(k=3)

# Step 7: Routing System
def route_question(question):
    """Route the question to either the vectorstore or web search."""
    router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

    Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

    routing_response = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions), HumanMessage(content=question)]
    )
    routing_decision = json.loads(routing_response.content)
    return routing_decision.get("datasource", "websearch")

# Step 8: Relevance Grader
def grade_relevance(document, question):
    """Assess relevance of a retrieved document to a user question."""
    doc_grader_prompt = f"""Here is the retrieved document: \n\n {document} \n\n 
    Here is the user question: \n\n {question}. 

    Assess whether the document contains at least some information that is relevant to the question. 
    Return JSON with a single key, binary_score, that is 'yes' or 'no'."""

    response = llm_json_mode.invoke(
        [SystemMessage(content="Assess document relevance"), HumanMessage(content=doc_grader_prompt)]
    )
    return json.loads(response.content)

# Step 9: Answer Generation
def generate_answer(context, question):
    """Generate an answer to a question using provided context."""
    rag_prompt = f"""You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:
    {context}

    Review the user question:
    {question}

    Provide a concise answer (3 sentences max):"""

    generation = llm.invoke([HumanMessage(content=rag_prompt)])
    return generation.content

# Step 10: Hallucination Grader
def grade_hallucination(facts, student_answer):
    """Assess whether a student's answer is grounded in provided facts."""
    hallucination_grader_prompt = f"""FACTS: \n\n {facts} \n\n STUDENT ANSWER: {student_answer}. 

    Score: binary_score ('yes' or 'no'), and explanation of reasoning."""

    response = llm_json_mode.invoke(
        [SystemMessage(content="Assess hallucination"), HumanMessage(content=hallucination_grader_prompt)]
    )
    return json.loads(response.content)






