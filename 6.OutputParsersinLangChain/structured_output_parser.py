from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define schemas
response_schemas = [  # Renamed variable to avoid naming collisions
    ResponseSchema(name="fact_1", description="fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = PromptTemplate(
    template="""
    Provide three interesting facts about the following topic: {topic}

    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser
response = chain.invoke({"topic": "Artificial Intelligence"})   

print(response)
print(type(response)) 
