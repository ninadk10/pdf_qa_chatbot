import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import tempfile
import os

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

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

    st.info(f"Step 3: Attempting to load Hugging Face Inference Client for model: `{MODEL_ID}`")
    try:
        client = InferenceClient(provider="featherless-ai", model=MODEL_ID, token=hf_token)
        st.success(f"Step 4: Successfully initialized Hugging Face Inference Client for `{MODEL_ID}`")
        return client
    except Exception as e:
        st.error(f"Step 5: Failed to initialize Hugging Face Inference Client for `{MODEL_ID}`: {e}")
        st.stop()

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

def retrieve_context(question, chunks, index):
    embedder = load_embedder()
    q_vec = embedder.encode([question])
    D, I = index.search(np.array(q_vec), k=3)
    return " ".join([chunks[i] for i in I[0]])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on PDF context."}
    ]

st.title("PDF Q&A Bot (Hugging Face Hosted Model)")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and indexing PDF..."):
        chunks, index = build_index(tmp_path)
    st.success("PDF processed. Ask your question!")

    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            context = retrieve_context(question, chunks, index)
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
            st.session_state.chat_history.append({"role": "user", "content": full_prompt})

            client = load_hf_client()
            try:
                response = client.conversational(messages=st.session_state.chat_history)
                assistant_reply = response["choices"][0]["message"]["content"]
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

            for msg in st.session_state.chat_history:
                if msg["role"] != "system":
                    st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

    os.remove(tmp_path)
