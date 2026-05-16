from dotenv import load_dotenv, parser
from google.auth import default
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


model = ChatGroq(
    model="llama-3.1-8b-instant",
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback as either positive or negative.")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative: \n {text} \n {format_instructions}",
    input_variables = ["text"],
    partial_variables = {"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "write an appropriate response to this positive feedback: \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "write an appropriate response to this negative feedback: \n {text}",
    input_variables = ["text"]
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not classify feedback sentiment")
)

chain = classifier_chain | branch_chain
text = """I had a great experience with your product! It exceeded my expectations and I will definitely recommend it to my friends."""
result = chain.invoke({"text": text})
print(result)

chain.get_graph().print_ascii()
