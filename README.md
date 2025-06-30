# PDF QA Chatbot

An interactive chatbot that allows users to upload PDF documents and ask natural language questions about their contents. It combines modern NLP techniques like **sentence embeddings**, **vector similarity search**, and **retrieval-based QA** into a streamlined and user-friendly application powered by **Streamlit**.

---

## Project Goals

This tool allows you to **query a document conversationally**, making it easier to extract answers without manual skimming or searching. This is especially useful for:

- Research papers
- Product manuals
- Contracts and legal documents
- Company reports

---

## How It Works

At a high level, the chatbot pipeline involves:

1. **PDF Parsing**  
   - Extract text from uploaded PDF files using `PyMuPDF` or `Fitz`.
   - Clean and normalize the text (remove empty lines, fix encodings, etc.).

2. **Text Chunking**  
   - Split the full document into smaller, semantically meaningful chunks.
   - Each chunk retains enough context to be independently understood.

3. **Embedding & Indexing**  
   - Use the `sentence-transformers/all-MiniLM-L6-v2` model to convert each chunk into a dense vector (embedding).
   - Store the vectors in a **FAISS** index for fast similarity search.

4. **Retrieval & Response**  
   - When a user asks a question, the system converts it to an embedding.
   - It retrieves the most relevant chunks via vector similarity.
   - A locally running Ollama model formulates an answer from the retrieved context.

---

## Tech Stack

| Component        | Library / Tool                      |
|------------------|-------------------------------------|
| Web UI           | Streamlit                           |
| PDF Parsing      | PyMuPDF / Fitz                      |
| Text Chunking    |                                     |
| Embedding Model  | sentence-transformers (MiniLM-L6-v2)|
| Vector Search    | FAISS                               |
| QA Logic         | Simple prompt + context retrieval   |

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf_qa_chatbot.git
cd pdf_qa_chatbot
