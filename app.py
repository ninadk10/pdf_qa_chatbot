import streamlit as st
import fitz, faiss, numpy as np, tempfile, os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

MODEL_ID = "tiiuae/falcon-7b-instruct"

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_hf_client():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        st.error("Hugging Face token not found.")
        st.stop()
    return InferenceClient(model=MODEL_ID, token=hf_token)

@st.cache_resource
def build_index(pdf_path):
    embedder = load_embedder()
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    emb = embedder.encode(chunks, batch_size=16, show_progress_bar=True)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    return chunks, idx

def retrieve_context(question, chunks, idx):
    vec = load_embedder().encode([question])
    _, I = idx.search(np.array(vec), k=3)
    return " ".join(chunks[i] for i in I[0])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Q&A Bot (Falcon‑7B‑Instruct)")
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    with st.spinner("Indexing PDF…"):
        chunks, idx = build_index(pdf_path)
    st.success("Done! Ask your questions below.")

    question = st.text_input("Your question:")
    if question:
        with st.spinner("Responding…"):
            ctx = retrieve_context(question, chunks, idx)
            prompt = f"Use the following context to answer the question:\n\n{ctx}\n\nQuestion: {question}\nAnswer:"

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            client = load_hf_client()
            resp = client.text_generation(prompt=prompt, max_new_tokens=256, temperature=0.7)
            reply = resp.strip()

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

    os.remove(pdf_path)
