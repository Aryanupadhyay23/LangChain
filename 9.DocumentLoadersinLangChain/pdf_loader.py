from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

loader = PyPDFLoader("file.pdf")
docs = loader.load()
print(docs)
print("\n")
print("-----------------------------")
print(docs[0])