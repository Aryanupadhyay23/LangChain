from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = PromptTemplate.from_template(
    "Generate five interesting facts about {topic}."
)

# Get API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "space exploration"})

print(result)

chain.get_graph().print_ascii()