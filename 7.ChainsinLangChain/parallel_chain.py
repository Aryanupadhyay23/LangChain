from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatGroq(
    model="llama-3.1-8b-instant",
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes on the following text \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Generate five short questions and answers based on the following notes \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n Notes: {notes} \n Quiz: {quiz}",
    input_variables = ["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Artificial Intelligence (AI) is a branch of computer science that focuses on creating machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI can be categorized into narrow AI, which is designed for specific tasks, and general AI, which has the potential to perform any intellectual task that a human can do. The development of AI has led to advancements in various fields such as healthcare, finance, and transportation, but it also raises ethical concerns regarding privacy, job displacement, and decision-making transparency."""

result = chain.invoke({"text":text})
print(result)

chain.get_graph().print_ascii()