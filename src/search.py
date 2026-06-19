from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


embeddings = SentenceTransformerEmbeddings()

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

query = "I forgot my password"

results = db.similarity_search(query, k=3)

print("\nQuery:", query)

print("\nRetrieved Documents:\n")

for doc in results:
    print("=" * 50)
    print(doc.metadata["source"])
    print(doc.page_content)