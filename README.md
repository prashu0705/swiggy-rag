# Swiggy Annual Report RAG Application

This repository contains a Retrieval-Augmented Generation (RAG) Question Answering system built on the Swiggy Annual Report (FY 2023-24). It allows users to ask natural language questions related to the report and receive accurate, context-grounded answers without hallucinating.

## Data Source
The application relies on the Swiggy Annual Report FY 2023-24 PDF, which is publicly available here:
[Swiggy Annual Report (FY 2023-24) PDF Link](https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf)

## Application Design
- **Document Processing**: The PDF is processed, and text is extracted using `pypdf`. The text is split into chunks of 1000 characters using Langchain's `RecursiveCharacterTextSplitter`.
- **Embeddings & Vector Store**: Document chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored locally using the `FAISS` vector database.
- **RAG QA System**: An open-source lightweight LLM (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) generates answers based strictly on the retrieved chunks using Langchain pipelines.

## Prerequisites
- Python 3.8+
- The `swiggy_annual_report.pdf` must be present in the root directory before running the system. (It is downloaded and renamed in this repo).

## Installation

1. Clone this repository and navigate to the directory.
2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Build the Vector Store
Before querying the system, you must build the FAISS vector index. This script will parse the PDF, generate embeddings, and store them locally.

```bash
python build_store.py
```
*Note: This will create a `faiss_index` folder in the root directory containing the vector store.*

### Step 2: Run the QA Application

You can interact with the QA system via the Command Line Interface (CLI).

**Option A: Ask a single query directly via argument**
```bash
python app.py --query "What is the standalone performance of Swiggy?"
```

**Option B: Interactive Mode**
Run the script without arguments to enter an interactive Q&A loop:
```bash
python app.py
```
You can type your questions and type `exit` or `quit` to stop the session. The system will display the **Final Answer** and the **Supporting Context (Top 3 Chunks)** for verification.