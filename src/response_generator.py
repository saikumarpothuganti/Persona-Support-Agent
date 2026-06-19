import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_response(query, context, persona):

    if persona == "Technical Expert":
        style = """
        Respond as a technical support engineer.
        Give detailed explanations.
        Include troubleshooting steps.
        Include root cause analysis.
        """

    elif persona == "Frustrated User":
        style = """
        Respond empathetically.
        Use simple language.
        Reassure the user.
        Give clear action steps.
        """

    else:
        style = """
        Respond as a business advisor.
        Focus on business impact.
        Keep the response concise.
        Avoid technical jargon.
        """

    prompt = f"""
    {style}

    Use ONLY the information below.

    Context:
    {context}

    User Question:
    {query}
    """

    response = model.generate_content(prompt)

    return response.text