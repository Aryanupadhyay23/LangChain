from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

loader = WebBaseLoader("https://www.aryanupadhyay.com/post/transformer-inference-step-by-step")
docs = loader.load()

# Initialize Gemini Flash model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

prompt = PromptTemplate(
    template = "Answer the following {question} from the following text:\n\n{text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"question": "How inference happens in transformer?", "text": docs[0].page_content})
print(result)
