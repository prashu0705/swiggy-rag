import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(pdf_path, index_path="faiss_index"):
    print(f"Loading document from {pdf_path}...")
    # 1. Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 2. Split text into chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate Embeddings using open-source model
    print("Generating embeddings and building vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # using CPU to save resources since it's lightweight
    )

    # 4. Store in Vector DB (FAISS)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the index locally
    vector_store.save_local(index_path)
    print(f"Vector store successfully built and saved to '{index_path}' directory.")

if __name__ == "__main__":
    pdf_file = "swiggy_annual_report.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: Could not find {pdf_file}")
    else:
        build_vector_store(pdf_file)
