import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables
load_dotenv()

def ingest_document(file_path, persist_directory="./db_free"):
    """
    Loads a PDF using PyMuPDF (better for tables), splits it into chunks, 
    generates embeddings using HF, and stores them in ChromaDB.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading document: {file_path} using PyMuPDF...")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    print(f"Splitting text into high-granularity chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Storing in ChromaDB at {persist_directory}...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Ensure directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="swiggy_annual_report"
    )
    
    print("Ingestion complete.")
    return vectorstore

if __name__ == "__main__":
    PDF_PATH = "Annual-Report-FY-2023-24 (1) (1).pdf"
    ingest_document(PDF_PATH)
