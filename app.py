import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Suppress HuggingFace warnings
import warnings
warnings.filterwarnings("ignore")

def setup_qa_system(index_path="faiss_index"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at '{index_path}'. Please run build_store.py first.")

    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("Loading LLM...")
    # Using TinyLlama as it is lightweight and runs well on CPU for RAG tasks
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu", # Force CPU
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Define a strict prompt to avoid hallucination
    prompt_template = """<|system|>
You are a helpful assistant that answers questions strictly based on the provided Swiggy Annual Report context. If the answer is not contained in the context, say "I don't know based on the provided document." Do not hallucinate or use outside knowledge.</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=question_answer_chain
    )

    return qa_chain

def main():
    parser = argparse.ArgumentParser(description="Swiggy Annual Report QA System")
    parser.add_argument("--query", type=str, help="Ask a question directly via CLI")
    args = parser.parse_args()

    try:
        qa_system = setup_qa_system()
    except Exception as e:
        print(f"Error initializing system: {e}")
        return

    print("\n--- Swiggy Annual Report QA System Ready ---")

    if args.query:
        answer_question(qa_system, args.query)
    else:
        print("Enter your questions below (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_input = input("\nQ: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break
                if not user_input.strip():
                    continue

                answer_question(qa_system, user_input)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

def answer_question(qa_system, query):
    print("\nSearching and generating answer...")
    result = qa_system.invoke({"input": query, "question": query})

    print("\n" + "="*50)
    print("Final Answer:")
    print("-" * 13)
    # The output from HF pipeline usually includes the prompt, we configured `return_full_text=False`
    print(result['answer'].strip())

    print("\n" + "="*50)
    print("Supporting Context (Top 3 Chunks):")
    print("-" * 34)
    if 'context' in result:
        for i, doc in enumerate(result['context']):
            print(f"\n[Chunk {i+1}]")
            print(doc.page_content.strip())
            print("-" * 34)

if __name__ == "__main__":
    main()
