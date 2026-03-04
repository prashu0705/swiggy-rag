import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from ingest import ingest_document

# Load env
load_dotenv()

# Page configuration
st.set_page_config(page_title="Swiggy Annual Report RAG", layout="wide")

# App header
st.title("Swiggy Annual Report FY 2023-24 Explorer (v5)")
st.markdown("Ask natural language questions about Swiggy's latest annual report.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    provider = st.radio(
        "Select AI Provider",
        ["Google Gemini (Cloud/Free Tier)", "Groq (High Speed - Free)", "OpenAI (GPT-4o/GPT-3.5)", "Ollama (Local/Offline)"],
        index=0,
        help="Gemini, Groq, and OpenAI are cloud providers. Ollama runs local."
    )
    
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = ""
    
    if provider == "Google Gemini (Cloud/Free Tier)":
        os.environ["GOOGLE_API_KEY"] = api_key if api_key else ""
        
        # Based on confirmed working models from your API key check
        model_name = st.selectbox(
            "Choose Gemini Model",
            ["gemini-flash-latest", "gemini-2.0-flash-exp", "gemini-1.5-flash-8b", "gemini-pro-latest"],
            index=0,
            help="gemini-flash-latest is confirmed to work with your account."
        )
    elif provider == "Groq (High Speed - Free)":
        model_name = st.selectbox(
            "Choose Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0
        )
    elif provider == "OpenAI (GPT-4o/GPT-3.5)":
        model_name = st.selectbox(
            "Choose OpenAI Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0
        )
    else:
        model_name = st.selectbox(
            "Choose Ollama Model",
            ["llama3.2:1b", "llama3.2:3b", "phi3:mini", "tinydolphin"],
            index=0
        )
        api_key = "local"

    st.divider()
    st.markdown("### About")
    st.info("This RAG system uses local embeddings (HuggingFace) for privacy and speed.")

# Main content
persist_dir = "./db_free"
pdf_path = "Annual-Report-FY-2023-24 (1) (1).pdf"

# Ingestion Logic
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    st.info("Vector database not found. Starting ingestion...")
    with st.spinner("Processing PDF... this takes ~1 min."):
        try:
            ingest_document(pdf_path, persist_dir)
            st.success("Ingestion complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# Load Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name="swiggy_annual_report"
)

# Query UI
user_query = st.text_input("Enter your question:", placeholder="e.g., What was the revenue growth for Swiggy in FY24?")

if user_query:
    with st.spinner(f"Using {model_name}..."):
        template = """### [SYSTEM]
You are a highly accurate financial document analysis assistant. Your goal is to provide precise numerical data and insights from the Swiggy Annual Report.

### [STRICT RULES]
1. **Source of Truth**: Use ONLY the provided context. Do NOT use external knowledge.
2. **Numerical Accuracy**: If a numerical value (Standalone vs Consolidated) is not explicitly in the text, say: "The value is not explicitly mentioned in the retrieved context." Do NOT guess or provide ranges.
3. **Chain-of-Thought (CoT)**: For complex queries (like subsidiary turnover), follow these steps:
   - Identify the specific table or section (e.g., 'Form AOC-1').
   - List the relevant entities and their values.
   - Verify the values against the context.
   - Summarize the final answer.
4. **Citations**: You MUST cite the exact page number for every fact or number you provide (e.g., "[Page 24]"). If the page number is not available, state the section title.
5. **No Hallucinations**: If the context says 'NIL' or is blank, report it as such. Do not assume '0' unless stated.

### [CONTEXT]
{context}

### [QUESTION]
{question}

### [CHAIN-OF-THOUGHT & FINAL ANSWER]
(Provide your internal verification steps first, then the final answer with page citations.)"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        try:
            if provider == "Google Gemini (Cloud/Free Tier)":
                # Using the exact string confirmed by the list_models diagnostic
                llm = ChatGoogleGenerativeAI(
                    model=model_name, 
                    temperature=0,
                    max_output_tokens=4096
                )
            elif provider == "Groq (High Speed - Free)":
                llm = ChatGroq(model=model_name, temperature=0)
            elif provider == "OpenAI (GPT-4o/GPT-3.5)":
                llm = ChatOpenAI(model=model_name, temperature=0)
            else:
                llm = OllamaLLM(model=model_name, temperature=0)

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )

            result = qa_chain.invoke({"query": user_query})
            
            st.markdown("### Final Answer")
            st.write(result["result"])

            with st.expander("Supporting Context"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page {doc.metadata.get('page', 'N/A')}:**")
                    st.write(doc.page_content)
                    st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check if the selected model is currently supported by your API key. Try 'gemini-flash-latest'.")
