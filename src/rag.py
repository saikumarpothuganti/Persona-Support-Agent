from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

docs = []

for file in os.listdir("data"):
    if file.endswith(".md"):
        loader = TextLoader(
            os.path.join("data", file),
            encoding="utf-8"
        )
        docs.extend(loader.load())

print("Documents Loaded:", len(docs))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("Chunks Created:", len(chunks))