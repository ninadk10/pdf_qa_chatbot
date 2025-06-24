import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import tempfile
import os

# Use a model that supports text-generation (Falcon-7B Instruct works)
MODEL_ID = "tiiuae/falcon-7b-instruct"

# Load sentence embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load Hugging Face inference client
@st.cache_resource
def load_hf_client():
    st.info("Loading Hugging Face Inference Client...")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        st.error("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        st.stop()

    try:
        client = InferenceClient(token=hf_token)
        st.success(f"Inference Client initialized for model `{MODEL_ID}`")
        return client
    except Exception as e:
        st.error(f"Failed to initialize client: {e}")
        st.stop()

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

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on PDF content."}
    ]

# Streamlit UI
st.title("📄 PDF Q&A Bot (Hugging Face Hosted Model)")
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
            client = load_hf_client()
            resp = client.text_generation(
                model=MODEL_ID,
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.7
            )
            reply = resp.strip()

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            for msg in st.session_state.chat_history:
                if msg["role"] != "system":
                    st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

    os.remove(tmp_path)
