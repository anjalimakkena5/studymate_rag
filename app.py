"""
app.py
Week 4: Simple Streamlit frontend for the StudyMate RAG system.

Run: streamlit run app.py
(Make sure the FastAPI backend is also running: uvicorn src.api:app --port 8000)
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="centered")

st.title("📚 StudyMate")
st.caption("Ask questions about your subject notes — answers are grounded in your own uploaded materials.")

# Keep a simple chat history in session state (in-memory only, resets on refresh)
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            with st.expander(f"📄 Sources ({len(entry['sources'])})"):
                for src in entry["sources"]:
                    st.markdown(f"**{src['source']}** (relevance score: {src['score']:.2f})")
                    st.caption(src["text_preview"] + "...")
        st.caption(f"⏱️ {entry['latency_ms']:.0f}ms")

# Input box
question = st.chat_input("Ask a question about your notes...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "top_k": 5},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                st.write(data["answer"])

                if data["sources"]:
                    with st.expander(f"📄 Sources ({len(data['sources'])})"):
                        for src in data["sources"]:
                            st.markdown(f"**{src['source']}** (relevance score: {src['score']:.2f})")
                            st.caption(src["text_preview"] + "...")

                st.caption(f"⏱️ {data['latency_ms']:.0f}ms")

                st.session_state.history.append(data)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the API. Make sure it's running: "
                    "`uvicorn src.api:app --port 8000`"
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend may be overloaded or still loading models.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Sidebar with reset option
with st.sidebar:
    st.header("About")
    st.write(
        "This is a RAG-based study assistant. It retrieves relevant chunks from "
        "your uploaded subject notes using hybrid search (dense + BM25 + re-ranking), "
        "then generates a grounded answer with citations."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
