import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import tempfile
import os

# Add your Hugging Face token here
HF_TOKEN = "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"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

# Load models
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_hf_client():
    return InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# Build FAISS index
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

# Retrieve context from FAISS
def retrieve_context(question, chunks, index):
    embedder = load_embedder()
    q_vec = embedder.encode([question])
    D, I = index.search(np.array(q_vec), k=3)
    return " ".join([chunks[i] for i in I[0]])

# Streamlit UI
st.title("PDF Q&A Bot (Hugging Face Hosted Model)")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Reading and indexing PDF..."):
        chunks, index = build_index(tmp_path)
    st.success("✅ PDF processed. Ask your question!")

    question = st.text_input("❓ Enter your question:")
    if question:
        with st.spinner("💬 Generating answer..."):
            context = retrieve_context(question, chunks, index)
            client = load_hf_client()
            prompt = f"[INST] Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer: [/INST]"
            response = client.text_generation(prompt, max_new_tokens=200, temperature=0.7)
            st.markdown("### ✅ Answer:")
            st.write(response.strip())

    os.remove(tmp_path)
