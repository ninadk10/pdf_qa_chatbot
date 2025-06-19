import streamlit as st
from qa_engine import QAEngine
from models.qa_model import LocalLLM

st.title("🧠 Smart PDF Q&A Bot")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question:")

if pdf_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    qa = QAEngine("temp.pdf")
    context = qa.query(question)

    llm = LocalLLM()
    answer = llm.answer(context, question)
    st.markdown("**Answer:** " + answer)
