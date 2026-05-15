from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a detailed explanation of the following topic: {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Write a five line summary of the following topic: {text}",
    input_variables = ["text"]
)

prompt1 = template1.format(topic="Artificial Intelligence")
response1 = model.invoke(prompt1)
prompt2 = template2.format(text=response1)
response2 = model.invoke(prompt2)

print("Detailed Explanation:\n", response1)
print("\nFive Line Summary:\n", response2)