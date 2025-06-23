import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import tempfile
import os


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Get the token from the environment variable
# hf_token = os.environ.get("HF_TOKEN")

# if hf_token is None:
#     raise ValueError("Hugging Face token not found. Please ensure it's in a .env file or set as an environment variable.")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Load models
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource # Use st.cache_resource to avoid re-initializing on every rerun
def load_hf_client():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        st.error("Hugging Face token not found. Please set the HF_TOKEN secret in Streamlit Cloud.")
        st.stop() # Stop the app if token is missing

    # --- CHANGE THIS LINE ---
    # Use a model known to be available on the Hugging Face Inference API for text generation
    # You can also omit 'model' here if you pass it directly in text_generation call,
    # but it's often good to initialize the client with a default model.
    model_id = "openai-community/gpt2" # Or "gpt2", "google/gemma-2b", etc.

    st.write(f"Attempting to load Hugging Face Inference Client for model: {model_id}") # For debugging

    try:
        client = InferenceClient(model=model_id, token=hf_token)
        st.success(f"Successfully initialized Hugging Face Inference Client for {model_id}") # For debugging
        return client
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face Inference Client for {model_id}: {e}")
        st.stop()


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

# test