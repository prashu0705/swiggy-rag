# Swiggy Annual Report RAG Application

An AI-powered Question Answering system built on Swiggy's Annual Report FY 2023-24 using Retrieval-Augmented Generation (RAG).

## Features
- **High-Granularity RAG**: Uses a refined parsing strategy to handle complex financial tables accurately.
- **Fast & Efficient**: Integrated with **Groq (Llama 3.3 70B)** for lightning-fast, high-capacity responses.
- **Improved Reasoning**: Implemented **Chain-of-Thought (CoT)** prompting for verifiable financial extraction.
- **Fact Verification**: Supports explicit **page citations** for every provided figure.
- **Flexible Providers**: Support for **Google Gemini**, **Groq**, **OpenAI**, and local **Ollama**.
- **Privacy First**: Uses local HuggingFace embeddings for data processing.

## Tech Stack
- **LLM Options**: Google Gemini, Ollama, OpenAI
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (Local)
- **Vector Store**: `ChromaDB` (Local)
- **Framework**: `LangChain`
- **Interface**: `Streamlit`

## Documentation Source
The data used in this application is the publicly available **Swiggy Annual Report FY 2023-24**.
- **Source Link**: [Swiggy Annual Report FY 2023-24](https://www.swiggy.com/corporate-governance)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd swiggy-rag-1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. Enter your question and get grounded answers!

## Deployment
For instructions on how to deploy this application to the cloud (e.g., Streamlit Community Cloud), refer to the [Deployment Guide](.agent/workflows/deploy.md).

---
*Built for the Swiggy RAG Assignment.*
