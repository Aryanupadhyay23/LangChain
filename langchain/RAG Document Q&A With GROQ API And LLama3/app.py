import streamlit as st
import os
import time
from dotenv import load_dotenv

# Core LangChain & Integration imports
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Corrected legacy chain imports (Requires: pip install langchain-classic)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Community integrations
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Define Chat Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. 
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        # --- FIX: Absolute Pathing ---
        # Gets the directory where this app.py file is actually saved
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_folder_path = os.path.join(current_dir, "research_papers")
        
        # Initialize Embeddings
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        
        # Load and Split Documents
        st.session_state.loader = PyPDFDirectoryLoader(pdf_folder_path)
        st.session_state.docs = st.session_state.loader.load()
        
        # --- FIX: IndexError Safety Check ---
        if not st.session_state.docs:
            st.error(f"❌ No PDFs found at: {pdf_folder_path}. Please check your folder.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        # Create Vector Store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("✅ Vector Database is ready!")

# Streamlit UI
st.title("RAG Document Q&A With GROQ & Llama3")

user_prompt = st.text_input("Enter your query from the research papers:")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please initialize the Document Embedding first.")
    else:
        # Build retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Execute query
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        
        st.write(f"⏱️ Response time: {time.process_time() - start:.2f} seconds")
        st.subheader("Answer")
        st.write(response['answer'])

        # Show source context
        with st.expander("Document Similarity Search (Context)"):
            for i, doc in enumerate(response['context']):
                st.info(f"**Source {i+1} Metadata:** {doc.metadata}")
                st.write(doc.page_content)
                st.write("---")
