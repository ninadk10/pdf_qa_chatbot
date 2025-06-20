import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import tempfile
import os

# Use a fast embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

# Build vector index from PDF
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

# Retrieve context
def retrieve_context(question, chunks, index):
    embedder = load_embedder()
    q_vec = embedder.encode([question])
    D, I = index.search(np.array(q_vec), k=3)
    return " ".join([chunks[i] for i in I[0]])

# Streamlit UI
st.title("⚡ PDF Q&A Bot (Optimized)")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        chunks, index = build_index(tmp_path)
    st.success("PDF processed. You can now ask questions.")

    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            context = retrieve_context(question, chunks, index)
            generator = load_generator()
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            answer = generator(prompt, max_new_tokens=100)[0]["generated_text"]
            st.markdown("### ✅ Answer:")
            st.write(answer.split("Answer:")[-1].strip())

    # Clean up temp file after use
    os.remove(tmp_path)
