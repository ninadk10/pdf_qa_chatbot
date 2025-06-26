import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import os
import requests

# Ollama config
OLLAMA_MODEL = "mistral"  # or llama3, phi3, etc.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Load sentence embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Build FAISS index from PDF
@st.cache_resource
def build_index(pdf_path):
    embedder = load_embedder()
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks, batch_size=16, show_progress_bar=True)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings))
    return chunks, index

# Retrieve relevant context
def retrieve_context(question, chunks, index):
    embedder = load_embedder()
    q_vec = embedder.encode([question])
    D, I = index.search(np.array(q_vec), k=3)
    return " ".join([chunks[i] for i in I[0]])

# Query Ollama locally
def ask_ollama(prompt, model=OLLAMA_MODEL):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"❌ Error querying Ollama: {e}"

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on PDF content."}
    ]

# Streamlit UI
st.title("📄 PDF Q&A Bot (Local Ollama Model)")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and indexing PDF..."):
        chunks, index = build_index(tmp_path)
    st.success("✅ PDF processed. Ask your question below.")

    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            context = retrieve_context(question, chunks, index)
            st.session_state.chat_history.append({"role": "user", "content": question})

            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
            reply = ask_ollama(prompt)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            for msg in st.session_state.chat_history:
                if msg["role"] != "system":
                    st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

    os.remove(tmp_path)
