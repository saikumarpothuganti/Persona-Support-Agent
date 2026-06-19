# Streamlit Community Cloud Deployment

## Required files

- `app.py` as the Streamlit entrypoint
- `requirements.txt` with only the runtime packages needed by the app
- `vector_db/` committed in the repository so Chroma can load the prebuilt index

## Streamlit Secrets

Create a `GEMINI_API_KEY` secret in Streamlit Community Cloud:

- Open the app settings in Streamlit Cloud
- Add the key `GEMINI_API_KEY`
- Paste the Gemini API key value

For local development, create `.streamlit/secrets.toml` with the same key, or use a `.env` file.

## Deploy steps

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create a new app from this repository.
3. Set the main file path to `app.py`.
4. Ensure Python 3.11 is selected.
5. Add `GEMINI_API_KEY` in Secrets.
6. Deploy.

## Notes

- The app caches the SentenceTransformer embedding model and the ChromaDB handle to reduce rerun overhead.
- If Chroma fails to load, rebuild `vector_db/` locally and commit the updated artifacts.
- If Gemini is not configured, the app now shows a clear error instead of failing silently.
