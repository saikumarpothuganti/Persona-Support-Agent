import os

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit is installed in deployment.
    st = None


def get_gemini_api_key():
    if st is not None:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                value = st.secrets["GEMINI_API_KEY"]

                if value:
                    return value
        except Exception:
            pass

    return os.getenv("GEMINI_API_KEY")
