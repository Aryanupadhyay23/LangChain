from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}'),
])

chat_history = []

with open('chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({'chat_history': chat_history, 'query': "How can I reset my password?"})

print("Prompt:", prompt)