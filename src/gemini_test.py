import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("AQ.Ab8RN6LMULM_dtpocNgNs3HNHsEkVIuESfd9-dLXBDhY5GR-Pw")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Say hello"
)

print(response.text)