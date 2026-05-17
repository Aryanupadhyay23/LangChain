from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

loader = DirectoryLoader("books", glob="*.pdf", show_progress=True, silent_errors=True, loader_cls=PyPDFLoader)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)