from dotenv import load_dotenv
import google.generativeai as genai

from src.settings import get_gemini_api_key

load_dotenv()

api_key = get_gemini_api_key()

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Configure Streamlit Secrets or a local .env file."
    )

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Say hello"
)

print(response.text)