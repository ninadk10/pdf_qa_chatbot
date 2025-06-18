import numpy as np
import faiss
import pymupdf
from sentence_transformers import SentenceTransformer


class QAEngine:
    def __init__(self, pdf_path):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = self._parse_pdf(pdf_path)
        self.embeddings = self.model.encode(self.chunks)
        self.index = faiss.IndexFlatL2(384)
        self.index.add(np.array(self.embeddings))

    def _parse_pdf(self, path):
        doc = fitz.open(path)
        full_text = " ".join([page.get_text() for page in doc])
        return [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    def query(self, question):
        q_vec = self.model.encode([question])
        D, I = self.index.search(np.array(q_vec), k=3)
        return " ".join([self.chunks[i] for i in I[0]])
