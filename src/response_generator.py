from dotenv import load_dotenv
import google.generativeai as genai

from src.settings import get_gemini_api_key

load_dotenv()

_MODEL = None


def _get_model():
    global _MODEL

    if _MODEL is None:
        api_key = get_gemini_api_key()

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Configure Streamlit Secrets or a local .env file."
            )

        genai.configure(api_key=api_key)
        _MODEL = genai.GenerativeModel("gemini-2.5-flash")

    return _MODEL


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

    response = _get_model().generate_content(prompt)

    return response.text