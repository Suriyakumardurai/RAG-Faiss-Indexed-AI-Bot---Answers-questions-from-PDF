import os
import io
import math
import faiss
import pickle
import fitz  # pymupdf
import pytesseract
import concurrent.futures
from PIL import Image
from typing import List, Tuple
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from groq import Groq

# ===============================
# CONFIG
# ===============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# thresholds
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB
MAX_WORKERS = min(8, (os.cpu_count() or 2))  # keep reasonable

# embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ===============================
# UTIL: text chunker (smart-ish)
# ===============================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunk text preserving sentence boundaries when possible.
    Fallback to simple sliding window when needed.
    """
    if not text:
        return []
    # naive sentence split on periods/newlines:
    import re
    sentences = re.split(r'(?<=[\.\?\!]\s)|\n+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += (" " if current else "") + s.strip()
        else:
            if current:
                chunks.append(current.strip())
            if len(s) > chunk_size:
                # break long sentence
                start = 0
                while start < len(s):
                    part = s[start:start+chunk_size]
                    chunks.append(part.strip())
                    start += chunk_size - overlap
                current = ""
            else:
                current = s.strip()
    if current:
        chunks.append(current.strip())
    # final normalization: ensure not too small fragments
    merged = []
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < chunk_size // 2 and i + 1 < len(chunks):
            merged.append((chunks[i] + " " + chunks[i+1]).strip())
            i += 2
        else:
            merged.append(chunks[i])
            i += 1
    return merged

# ===============================
# PDF PAGE PROCESSING (text + images + OCR)
# ===============================
def process_page(doc_path: str, page_number: int, ocr: bool = True) -> str:
    """
    Extract text from one page and OCR any images on that page.
    Returns a combined text string for that page.
    """
    page_text = ""
    try:
        with fitz.open(doc_path) as doc:
            page = doc.load_page(page_number)
            # primary text extraction
            txt = page.get_text("text") or ""
            page_text += txt + "\n"

            if ocr:
                # extract images from the page
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    try:
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        # rudimentary check: skip tiny images
                        if image.width < 20 or image.height < 20:
                            continue
                        # OCR the image
                        try:
                            ocr_result = pytesseract.image_to_string(image)
                            if ocr_result and ocr_result.strip():
                                page_text += "\n[OCR_IMAGE_TEXT]\n" + ocr_result.strip() + "\n"
                        except Exception as e:
                            # OCR failure should not break flow
                            page_text += f"\n[OCR_ERROR image {img_index}]: {e}\n"
                    except Exception as ie:
                        page_text += f"\n[IMG_DECODE_ERROR image {img_index}]: {ie}\n"
    except Exception as e:
        page_text += f"\n[PAGE_ERROR page {page_number}]: {e}\n"

    return page_text.strip()

def process_page_batch(doc_path: str, page_range: Tuple[int, int]) -> List[str]:
    """
    Process a batch (inclusive start, exclusive end) and return page texts (one per page)
    """
    start, end = page_range
    results = []
    for p in range(start, end):
        results.append(process_page(doc_path, p, ocr=True))
    return results

# ===============================
# VECTOR STORE (text chunks)
# ===============================
class VectorStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.index = None
        self.dimension = None

    def build(self, chunks: List[str], add: bool = False):
        """
        Build or update a FAISS index from chunks.
        If add=True and an index exists, it will add embeddings to existing index.
        """
        if not chunks:
            return
        embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        if self.index is None or not add:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            self.chunks = list(chunks)
        else:
            # append embeddings and chunks
            self.index.add(embeddings)
            self.chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.index is None or len(self.chunks) == 0:
            return []
        query_emb = embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        return results

    def save(self, path: str = "vectorstore"):
        os.makedirs(path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str = "vectorstore") -> bool:
        faiss_path = os.path.join(path, "faiss.index")
        chunks_path = os.path.join(path, "chunks.pkl")
        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(faiss_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            # set dimension if possible
            if self.index is not None:
                try:
                    self.dimension = self.index.d
                except Exception:
                    self.dimension = None
            return True
        return False

vector_store = VectorStore()
vector_store.load()

# ===============================
# LLM CALL (Groq)
# ===============================
def generate_answer(query: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the following context. If the context doesn't contain the answer say "I don't know from the provided documents."

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
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI()
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
    if not os.path.exists("templates/index.html"):
        return HTMLResponse("<h3>Upload endpoint is /upload_pdf/</h3>")
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ===============================
# ROUTES
# ===============================
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    # Save uploaded file
    file_path = f"./uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    size = len(contents)
    # open with fitz to get page count
    try:
        doc = fitz.open(file_path)
        page_count = doc.page_count
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDF or read error: {e}")

    # If large file: multithread page processing
    page_texts = []
    if size > LARGE_FILE_THRESHOLD and page_count > 1:
        # compute batch ranges per worker
        workers = min(MAX_WORKERS, page_count)
        pages_per_worker = math.ceil(page_count / workers)
        ranges = []
        for i in range(workers):
            start = i * pages_per_worker
            end = min(start + pages_per_worker, page_count)
            if start < end:
                ranges.append((start, end))

        # process concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_page_batch, file_path, r) for r in ranges]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    page_texts.extend(res)
                except Exception as e:
                    page_texts.append(f"[BATCH_ERROR] {e}")
    else:
        # single-threaded page processing
        for p in range(page_count):
            page_texts.append(process_page(file_path, p, ocr=True))

    # Combine all page texts and chunk them
    # filter empty pages and add page boundary markers for provenance
    annotated_pages = []
    for i, pt in enumerate(page_texts):
        if not pt or len(pt.strip()) == 0:
            continue
        annotated_pages.append(f"[PAGE {i+1}/{page_count}]\n{pt}")

    # Chunk each page separately to keep provenance
    all_chunks = []
    for ptext in annotated_pages:
        chunks = chunk_text(ptext, chunk_size=700, overlap=100)  # bigger chunk size for better context
        all_chunks.extend(chunks)

    # Build or update vector store
    # If previously loaded vectorstore exists, we'll append (incremental)
    add_to_existing = vector_store.index is not None and len(vector_store.chunks) > 0
    vector_store.build(all_chunks, add=add_to_existing)
    vector_store.save()

    return {"message": f"PDF {file.filename} processed. Pages: {page_count}. Chunks added: {len(all_chunks)}."}

@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    results = vector_store.retrieve(query, top_k=4)
    if not results:
        return {"answer": "No data available. Upload a PDF first."}
    answer = generate_answer(query, results)
    return {"answer": answer, "contexts": results}

# ===============================
# RUN (for testing)
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
