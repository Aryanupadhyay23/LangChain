from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatOllama(model="tinyllama")

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    print("AI:", response.content)

print("Chat history:", chat_history)