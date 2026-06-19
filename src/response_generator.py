import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)


def generate_response(query, persona, context):
    prompt = f"""
Persona: {persona}

Context:
{context}

User Question:
{query}

Provide a helpful response according to the persona.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text
