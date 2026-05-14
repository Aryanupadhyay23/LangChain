from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="gemma:2b")

# detailed way
template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

# fill the values of the placeholders
prompt = template2.invoke({'name': 'Aryan'})

result = model.invoke(prompt)

print(result)