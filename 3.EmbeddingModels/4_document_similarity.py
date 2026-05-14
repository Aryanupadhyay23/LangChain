from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
documents = ["The capital of France is Paris.", "The capital of Germany is Berlin."]
query = "What is the capital of France?"
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)
print(similarity_scores)