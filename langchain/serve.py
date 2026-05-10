# Import FastAPI framework
from fastapi import FastAPI

# Import LangChain prompt template
from langchain_core.prompts import ChatPromptTemplate

# Import output parser to convert model response into string
from langchain_core.output_parsers import StrOutputParser

# Import Groq LLM integration
from langchain_groq import ChatGroq

# Import os module for environment variables
import os

# Import utility to expose LangChain chains as API routes
from langserve import add_routes

# Import dotenv loader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# Define system instruction template
system_template = "Translate the following into {language}: "

# Create chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}")
    ]
)

# Create output parser
parser = StrOutputParser()

# Create LangChain pipeline
chain = prompt_template | model | parser

# Create FastAPI application
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

# Add API route for the chain
add_routes(
    app,
    chain,
    path="/chain"
)

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)