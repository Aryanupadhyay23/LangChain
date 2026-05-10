import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions. Please respond in a concise manner."),
        ("user", "Question: {question}")
    ]
)

# Streamlit app
st.title("Simple GenAI App using gemma:2b")
question = st.text_input("Enter your question: ")

# Ollama model
llm = Ollama(model="gemma:2b")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if question:
    response = chain.invoke({"question": question})
    st.write(response)