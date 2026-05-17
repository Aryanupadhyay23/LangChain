from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("file.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="")

result = splitter.split_documents(docs)
for i, doc in enumerate(result):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 20)

