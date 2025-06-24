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

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Load models
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_hf_client():
    st.info("Step 1: Inside load_hf_client.")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        st.error("Hugging Face token not found. Please set the HF_TOKEN secret in Streamlit Cloud.")
        st.stop()
    else:
        st.info("Step 2: HF_TOKEN found in environment.")
        # st.write(f"HF Token length: {len(hf_token)}") # You can print length to confirm it's not empty, but don't print actual token!

    model_id = MODEL_ID # Still stick with gpt2 for now

    st.info(f"Step 3: Attempting to load Hugging Face Inference Client for model: `{model_id}`")

    try:
        client = InferenceClient(model=model_id, token=hf_token)
        st.success(f"Step 4: Successfully initialized Hugging Face Inference Client for `{model_id}`")
        return client
    except Exception as e:
        st.error(f"Step 5: Failed to initialize Hugging Face Inference Client for `{model_id}`: {e}")
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
            response = client.conversational(prompt, max_new_tokens=200, temperature=0.7)
            st.markdown("### ✅ Answer:")
            st.write(response.strip())
    os.remove(tmp_path)

# test