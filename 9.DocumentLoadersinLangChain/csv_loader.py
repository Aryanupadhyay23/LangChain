from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("csv_file.csv")

docs = loader.load()
print(docs)
print(len(docs))
print(docs[0])
print(docs[0].page_content)