from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="gemma:2b")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result))
print(messages)