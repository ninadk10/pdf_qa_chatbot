import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import tempfile
import os

# Load fast embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load FLAN-T5 for Q&A
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

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

# Retrieve top chunks from index
def retrieve_context(question, chunks, index):
    embedder = load_embedder()
    q_vec = embedder.encode([question])
    D, I = index.search(np.array(q_vec), k=3)
    return " ".join([chunks[i] for i in I[0]])

# Streamlit UI
st.title("PDF Q&A Chatbot")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and indexing PDF..."):
        chunks, index = build_index(tmp_path)
    st.success("✅ PDF ready. Ask your question!")

    question = st.text_input("❓ Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            context = retrieve_context(question, chunks, index)
            generator = load_generator()
            prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
            result = generator(prompt, max_new_tokens=100)[0]['generated_text']
            st.markdown("### ✅ Answer:")
            st.write(result.strip())

    os.remove(tmp_path)
