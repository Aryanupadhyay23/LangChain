from langchain_community.document_loaders import TextLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate



loader = TextLoader("speech.txt")

# Initialize Gemini Flash model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

prompt = PromptTemplate(
    template = "Summarize the following text:\n\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

docs = loader.load()

# print(docs)
# print(type(docs))
# print(len(docs))
# print(docs[0])
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser
print(chain.invoke({"text": docs[0].page_content}))

