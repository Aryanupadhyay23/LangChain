from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

model = ChatOpenAI(model = "gpt-4", temperature=0.2, max_completion_tokens=30)
result = model.invoke("What is the capital of India?")

print(result.content)
