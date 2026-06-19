from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import os


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


docs = []

for file in os.listdir("data"):
    if file.endswith(".md"):
        loader = TextLoader(
            os.path.join("data", file),
            encoding="utf-8"
        )
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings()

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)

print("Vector Database Created Successfully!")
print("Chunks Stored:", len(chunks))