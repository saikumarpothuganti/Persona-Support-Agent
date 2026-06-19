def generate_handoff(
    persona,
    query,
    results
):

    return {
        "persona": persona,
        "issue": query,
        "documents_used": [
            doc.metadata["source"]
            for doc in results
        ],
        "attempted_steps": [
            "Knowledge base search",
            "AI generated response"
        ],
        "recommendation":
            "Escalate to human support agent"
    }