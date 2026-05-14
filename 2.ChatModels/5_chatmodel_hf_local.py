from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task="text-generation",
    pipeline_kwargs=(
        {"temperature": 0.5, "max_new_tokens": 100}
    )
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)