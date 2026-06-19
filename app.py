import streamlit as st

from src.persona_detector import detect_persona
from src.response_generator import generate_response
from src.escalation import should_escalate
from src.handoff import generate_handoff

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


st.set_page_config(
    page_title="Persona-Aware Support Agent",
    layout="wide"
)

st.title("🤖 Persona-Aware Customer Support Agent")

query = st.text_area(
    "Enter your support query",
    height=120
)

if st.button("Submit"):

    if query:

        persona = detect_persona(query)

        embeddings = SentenceTransformerEmbeddings()

        db = Chroma(
            persist_directory="vector_db",
            embedding_function=embeddings
        )

        results = db.similarity_search(query, k=5)

        unique_results = []
        seen_sources = set()

        for doc in results:
            source = doc.metadata["source"]

            if source not in seen_sources:
                unique_results.append(doc)
                seen_sources.add(source)

        results = unique_results[:3]

        st.subheader("Detected Persona")
        st.success(persona)

        st.subheader("Retrieved Sources")

        for doc in results:
            st.write(doc.metadata["source"])

        if should_escalate(query, results):

            st.error("Escalation Required")

            summary = generate_handoff(
                persona,
                query,
                results
            )

            st.subheader("Human Handoff Summary")
            st.json(summary)

        else:

            context = "\n".join(
                [doc.page_content for doc in results]
            )

            response = generate_response(
                query,
                context,
                persona
            )

            st.subheader("Generated Response")
            st.write(response)