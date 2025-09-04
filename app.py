import os
import faiss
import torch
from pypdf import PdfReader
from typing import List
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ===============================
# CONFIG
# ===============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# ===============================
# STEP 1: PDF TEXT EXTRACTION
# ===============================
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ===============================
# STEP 2: CHUNKING
# ===============================
def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ===============================
# STEP 3: EMBEDDINGS + FAISS
# ===============================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import pickle

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.dimension = None

    def build(self, chunks: List[str]):
        self.chunks = chunks
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k=3) -> List[str]:
        if self.index is None:
            return []
        query_emb = embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in indices[0]]

    def save(self, path="vectorstore"):
        os.makedirs(path, exist_ok=True)
        # save faiss index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        # save chunks
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path="vectorstore"):
        faiss_path = os.path.join(path, "faiss.index")
        chunks_path = os.path.join(path, "chunks.pkl")
        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(faiss_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            return True
        return False


vector_store = VectorStore()
vector_store.load() 

# ===============================
# STEP 4: LLM CALL (Groq)
# ===============================
def generate_answer(query: str, retrieved_chunks: List[str]) -> str:
    context = "\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the following context:

Context:
{context}

Question: {query}
Answer succinctly:"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for RAG Q&A."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_completion_tokens=800
    )
    return resp.choices[0].message.content.strip()

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI()
# mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
# ===============================
# BACKEND ROUTES
# ===============================
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    raw_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(raw_text)
    vector_store.build(chunks)
    vector_store.save()  # 🔥 persist to disk
    return {"message": f"PDF {file.filename} processed & stored. {len(chunks)} chunks saved."}

@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    results = vector_store.retrieve(query, top_k=3)
    if not results:
        return {"answer": "No data available. Upload a PDF first."}
    answer = generate_answer(query, results)
    return {"answer": answer, "contexts": results}

# ===============================
# RUN (for testing)
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
