import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Please answer the user questions clearly."
        ),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, model, temperature, max_tokens):

    # Load Ollama model
    llm = Ollama(
        model=model,
        temperature=temperature
    )

    # Output parser
    output_parser = StrOutputParser()

    # Create chain
    chain = prompt | llm | output_parser

    # Generate answer
    answer = chain.invoke(
        {
            "question": question
        }
    )

    return answer


# Streamlit UI
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖"
)

st.title("Simple Q&A Chatbot With Ollama")

# Sidebar settings
st.sidebar.header("Settings")

# Select local Ollama model
model = st.sidebar.selectbox(
    "Select Ollama Model",
    [
        "llama3",
        "mistral",
        "gemma:2b",
        "phi3"
    ]
)

# Temperature
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

# Max tokens
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=500,
    value=150
)

# User input
user_input = st.text_input("Ask a question")

# Generate response
if user_input:

    with st.spinner("Generating response..."):

        try:
            response = generate_response(
                user_input,
                model,
                temperature,
                max_tokens
            )

            st.success("Response Generated")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")