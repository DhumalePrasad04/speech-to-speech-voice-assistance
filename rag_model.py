import fitz  # PyMuPDF for PDF parsing
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_ollama.chat_models import ChatOllama

# Step 1: Load Data (from PDF or web)
import fitz  # PyMuPDF for PDF parsing

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF including paragraphs, tables, and images."""
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract all the text from the page (paragraphs, headings)
        text += page.get_text("text") + "\n"

        # Extract structured data (tables, lists, and other elements)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # This is a text block (paragraph)
                if 'text' in block:
                    text += block['text'] + "\n"
            elif block['type'] == 1:  # This is an image or other non-text element, skip it
                continue
            elif block['type'] == 2:  # This could be a table or complex element
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                text += "\n"

    doc.close()
    return text



def load_data_from_url(url):
    """Load document from a URL and extract text."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return text

# Step 2: Chunk Text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Step 3: Convert Chunks to LangChain Documents
def create_documents_from_chunks(chunks, metadata=None):
    """Create LangChain Documents from text chunks."""
    return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

# Step 4: Embed and Store Data in Vector DB
def embed_and_store_documents(documents, vectorstore_path):
    """Embed documents and store them in a vector database."""
    embedding_model = OllamaEmbeddings(model="llama3.2:1b")  # Use Ollama model for embeddings
    # Create and persist vectorstore using Chroma
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=vectorstore_path)
    # Chroma automatically persists when setting the persist_directory
    return vectorstore

# Step 5: Initialize the RAG Pipeline
def initialize_rag_pipeline(vectorstore_path):
    """Create a RAG pipeline with retrieval and generation."""
    # Initialize Chroma with the specified embedding function
    embedding_model = OllamaEmbeddings(model="llama3.2:1b")
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model="llama3.2:1b", temperature=0)  # Use ChatOllama with Llama 3.2 1B model

    # Setup the RetrievalQA chain with retriever and LLM
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Step 6: Answer Query
def answer_query(rag_pipeline, question):
    """Process the query through RAG pipeline."""
    return rag_pipeline.run(question)

# Main Code to Execute Pipeline
if __name__ == "__main__":
    pdf_path = "reva.txt"  # Path to your PDF document
    vectorstore_path = "./vectorbase"  # Directory to store vector data
    urls = [
        "https://www.reva.edu.in/",

    ]

    # Load PDF or Web data
    text = ""
    if pdf_path:
        text = extract_text_from_pdf(pdf_path)
    else:
        for url in urls:
            text += load_data_from_url(url)

    # Chunk the text
    chunks = chunk_text(text)
    documents = create_documents_from_chunks(chunks, metadata={"source": pdf_path})

    # Embed and Store Data
    vectorstore = embed_and_store_documents(documents, vectorstore_path)

    # Initialize RAG Pipeline
    rag_pipeline = initialize_rag_pipeline(vectorstore_path)

    # Example Query
    question = "contact of reva university?"
    answer = answer_query(rag_pipeline, question)
    print(f"Answer: {answer}")
