from persona_detector import detect_persona
from response_generator import generate_response
from escalation import should_escalate
from handoff import generate_handoff

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


query = input("Ask: ")

persona = detect_persona(query)

embeddings = SentenceTransformerEmbeddings()

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

# Retrieve documents
results = db.similarity_search(query, k=5)

# Remove duplicate sources
unique_results = []
seen_sources = set()

for doc in results:
    source = doc.metadata["source"]

    if source not in seen_sources:
        unique_results.append(doc)
        seen_sources.add(source)

results = unique_results[:3]

# Escalation Check
if should_escalate(query, results):

    print("\n==============================")
    print("ESCALATION REQUIRED")
    print("==============================\n")

    summary = generate_handoff(
        persona,
        query,
        results
    )

    print(summary)

else:

    context = "\n".join(
        [doc.page_content for doc in results]
    )

    answer = generate_response(
        query,
        context,
        persona
    )

    print("\n==============================")
    print("PERSONA DETECTED")
    print("==============================")
    print(persona)

    print("\n==============================")
    print("RETRIEVED SOURCES")
    print("==============================")

    for doc in results:
        print(doc.metadata["source"])

    print("\n==============================")
    print("AI RESPONSE")
    print("==============================")
    print(answer)