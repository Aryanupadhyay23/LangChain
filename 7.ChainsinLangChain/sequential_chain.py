from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

Prompt1 = PromptTemplate(
    template = "Generate a detailed report on {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate the five most important points from the following report: {text}",
    input_variables = ["text"]
)

# Initialize model
model = ChatGroq(
    model="llama-3.1-8b-instant",
)

parser = StrOutputParser()

chain = Prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "the impact of artificial intelligence on society"})
print(result)