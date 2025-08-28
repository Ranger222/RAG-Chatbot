import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import requests
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("FAISS import failed, ensure faiss-cpu is installed") from e

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("[WARN] MISTRAL_API_KEY not set. Set it in environment before running in production.")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini API configured successfully.")
elif not GEMINI_API_KEY:
    print("[WARN] GEMINI_API_KEY not set. Gemini provider will not be available.")

MISTRAL_CHAT_MODEL = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-large-latest")
MISTRAL_EMBED_MODEL = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_FALLBACK_MODEL = os.environ.get("MISTRAL_FALLBACK_MODEL", "mistral-small-latest")

BASE_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / "storage"
FILES_DIR = STORAGE_DIR / "files"  # served statically
META_DIR = STORAGE_DIR / "meta"     # per-doc metadata
INDEX_DIR = STORAGE_DIR / "indexes" # per-doc FAISS

for d in (FILES_DIR, META_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

DOCS_LIST_PATH = STORAGE_DIR / "documents.json"  # simple catalog

# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------
class Document(BaseModel):
    id: str
    name: str
    uploadedOn: str
    size: int
    type: str = "pdf"
    fileUrl: Optional[str] = None

class DocumentsResponse(BaseModel):
    documents: List[Document]

class UploadResponse(BaseModel):
    document: Document

class ChatRequest(BaseModel):
    docId: str
    message: str
    max_chunks: int = 4
    provider: str = "mistral"  # "mistral" or "gemini"

class ChatReference(BaseModel):
    docId: str
    page: int
    text: Optional[str] = None

class WebSource(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    reference: Optional[ChatReference] = None
    origin: str = "document"  # document | web | hybrid
    sources: Optional[List[WebSource]] = None

# ----------------------------------------------------------------------------
# Helpers: persistence
# ----------------------------------------------------------------------------

def load_documents() -> List[Document]:
    if DOCS_LIST_PATH.exists():
        data = json.loads(DOCS_LIST_PATH.read_text())
        return [Document(**d) for d in data]
    return []

def save_documents(docs: List[Document]):
    DOCS_LIST_PATH.write_text(json.dumps([d.model_dump() for d in docs], indent=2))

# ----------------------------------------------------------------------------
# Helpers: PDF parsing
# ----------------------------------------------------------------------------

def extract_pdf_pages_text(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for i, p in enumerate(reader.pages):
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

# ----------------------------------------------------------------------------
# Helpers: Embeddings with Mistral
# ----------------------------------------------------------------------------

from duckduckgo_search import DDGS
EMBEDDING_DIM_CACHE: Optional[int] = None

# --- Embedding batching and splitting controls (char-based token estimate) ---
TOKEN_EST_CHARS_PER_TOKEN = 4
EMBED_BATCH_MAX_TOKENS = int(os.environ.get("EMBED_BATCH_MAX_TOKENS", "6000"))  # conservative cap per request
EMBED_BATCH_MAX_ITEMS = int(os.environ.get("EMBED_BATCH_MAX_ITEMS", "32"))      # limit number of strings per request
ITEM_MAX_TOKENS = int(os.environ.get("EMBED_ITEM_MAX_TOKENS", "1200"))          # per-item soft cap before splitting
ITEM_CHUNK_OVERLAP_TOKENS = int(os.environ.get("EMBED_ITEM_CHUNK_OVERLAP_TOKENS", "100"))


def estimate_tokens(s: str) -> int:
    return max(1, len(s) // TOKEN_EST_CHARS_PER_TOKEN)


def split_text_by_chars(s: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    if not s:
        return [s]
    max_chars = max_tokens * TOKEN_EST_CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * TOKEN_EST_CHARS_PER_TOKEN
    chunks: List[str] = []
    n = len(s)
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunk = s[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        # step forward with overlap
        next_start = end - overlap_chars
        start = next_start if next_start > start else end
    return chunks


def _embed_texts_gemini(items: List[str]) -> np.ndarray:
    """Embed texts using Gemini embedding model."""
    import google.generativeai as genai
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=502, detail="No embedding service available: Mistral and Gemini API keys missing")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        embeddings = []
        for text in items:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            embeddings.append(np.array(result['embedding'], dtype=np.float32))
        
        if embeddings:
            return np.stack(embeddings)
        else:
            return np.zeros((0, 768), dtype=np.float32)  # Gemini embedding dimension
            
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini embeddings error: {str(e)}")

def _embed_texts_batched(items: List[str]) -> np.ndarray:
    """Embed a list of reasonably sized strings by batching to stay under API limits."""
    global EMBEDDING_DIM_CACHE
    if not items:
        return np.zeros((0, 0), dtype=np.float32)

    # Use Gemini embeddings if Mistral key is not available
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
        return _embed_texts_gemini(items)

    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def flush(batch: List[str], out: List[np.ndarray]):
        if not batch:
            return
        payload = {"model": MISTRAL_EMBED_MODEL, "input": batch}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 300:
            # surface the error for front-end to display
            raise HTTPException(status_code=502, detail=f"Mistral embeddings error: {r.text}")
        data = r.json()
        vecs = [np.array(item["embedding"], dtype=np.float32) for item in data.get("data", [])]
        out.extend(vecs)

    all_vectors: List[np.ndarray] = []
    batch: List[str] = []
    toks_sum = 0
    count = 0

    for t in items:
        t_toks = estimate_tokens(t)
        # If adding this would exceed either limit, flush first
        if batch and (toks_sum + t_toks > EMBED_BATCH_MAX_TOKENS or count >= EMBED_BATCH_MAX_ITEMS):
            flush(batch, all_vectors)
            batch = []
            toks_sum = 0
            count = 0
        batch.append(t)
        toks_sum += t_toks
        count += 1

    # final flush
    flush(batch, all_vectors)

    if not all_vectors:
        return np.zeros((0, 0), dtype=np.float32)

    EMBEDDING_DIM_CACHE = len(all_vectors[0])
    return np.vstack(all_vectors)


# Replace original embed_texts with batching + mean-pooling for oversize items
# ----------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    global EMBEDDING_DIM_CACHE
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    # First, split oversize items and mean-pool their embeddings
    pooled_vectors: Dict[int, Optional[np.ndarray]] = {}
    short_texts: List[str] = []
    short_indices: List[int] = []

    for idx, t in enumerate(texts):
        if estimate_tokens(t) > ITEM_MAX_TOKENS:
            parts = split_text_by_chars(t, max_tokens=ITEM_MAX_TOKENS, overlap_tokens=ITEM_CHUNK_OVERLAP_TOKENS)
            part_vecs = _embed_texts_batched(parts)
            if part_vecs.size == 0:
                pooled_vectors[idx] = None  # fill later once we know dim
            else:
                pooled_vectors[idx] = part_vecs.mean(axis=0).astype(np.float32)
        else:
            short_indices.append(idx)
            short_texts.append(t)

    # Embed remaining short items in batches
    short_vecs = _embed_texts_batched(short_texts) if short_texts else np.zeros((0, 0), dtype=np.float32)

    # Determine embedding dimension to fill missing entries if any
    dim: int = 0
    if short_vecs.size > 0:
        dim = short_vecs.shape[1]
        EMBEDDING_DIM_CACHE = dim
    elif EMBEDDING_DIM_CACHE:
        dim = EMBEDDING_DIM_CACHE
    else:
        probe = _embed_texts_batched([" "])
        if probe.size > 0:
            dim = probe.shape[1]
            EMBEDDING_DIM_CACHE = dim

    # Assemble in original order
    assembled: List[np.ndarray] = []
    short_ptr = 0
    for i in range(len(texts)):
        if i in pooled_vectors:
            vec = pooled_vectors[i]
            if vec is None:
                if dim <= 0:
                    # no dimension known; return empty to avoid downstream errors
                    return np.zeros((0, 0), dtype=np.float32)
                vec = np.zeros((dim,), dtype=np.float32)
            assembled.append(vec)
        else:
            assembled.append(short_vecs[short_ptr])
            short_ptr += 1

    if not assembled:
        return np.zeros((0, 0), dtype=np.float32)

    if assembled and assembled[0].size > 0:
        EMBEDDING_DIM_CACHE = assembled[0].shape[0]
    return np.vstack(assembled)

# ----------------------------------------------------------------------------
# Helpers: FAISS per-document index
# ----------------------------------------------------------------------------

def index_paths(doc_id: str) -> Dict[str, Path]:
    return {
        "faiss": INDEX_DIR / f"{doc_id}.index",
        "mapping": INDEX_DIR / f"{doc_id}.mapping.json",
    }


def build_index_for_document(doc_id: str, page_texts: List[str]):
    # Create passages (here, whole page as one chunk). Could be further chunked if long.
    passages = []
    for i, t in enumerate(page_texts, start=1):
        if not t:
            continue
        # Basic cleanup and truncate long pages
        snippet = t.strip().replace("\n", " ")
        passages.append({"page": i, "text": snippet})

    if not passages:
        # still create empty mapping/index files
        paths = index_paths(doc_id)
        json.dump(passages, open(paths["mapping"], "w"))
        index = faiss.IndexFlatIP(1)
        faiss.write_index(index, str(paths["faiss"]))
        return

    texts = [p["text"] for p in passages]
    embs = embed_texts(texts)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    paths = index_paths(doc_id)
    faiss.write_index(index, str(paths["faiss"]))
    with open(paths["mapping"], "w") as f:
        json.dump(passages, f)


def search_doc(doc_id: str, query: str, top_k: int = 4):
    paths = index_paths(doc_id)
    if not paths["faiss"].exists() or not paths["mapping"].exists():
        raise HTTPException(status_code=404, detail="Index not found for document")
    index = faiss.read_index(str(paths["faiss"]))
    passages = json.load(open(paths["mapping"]))
    qvec = embed_texts([query]).astype(np.float32)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, min(top_k, len(passages)))
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        p = passages[idx]
        results.append({"score": float(score), **p})
    return results

# ----------------------------------------------------------------------------
# Helpers: Gemini chat completion
# ----------------------------------------------------------------------------

def chat_with_gemini(question: str, contexts: List[Dict[str, Any]], web_results: Optional[List[Dict[str, str]]] = None) -> str:
    if not genai or not GEMINI_API_KEY:
        return "Gemini API is not configured. Please check your GEMINI_API_KEY environment variable."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Build context
        context_text = "\n\n".join([f"[Doc Page {c['page']}] {c['text'][:1200]}" for c in contexts[:4]])
        web_text = ""
        if web_results:
            web_text = "\n\n".join([f"[Web {i+1}] {w.get('title','')} — {w.get('snippet','')[:400]}\nURL: {w.get('url','')}" for i, w in enumerate(web_results[:3])])
        
        combined_context = context_text
        if web_text:
            combined_context = (context_text + "\n\nExternal Web Results:\n" + web_text).strip()
        
        system_prompt = (
            "You are a helpful assistant that answers questions about the user's document. "
            "Use the provided context passages to answer succinctly. If unsure, say you don't know. "
            "If external web results are provided, you may use them, but clearly indicate sources and do not claim they are in the document. "
            "If the user greets with something like 'hi', 'hello', or 'hey', respond briefly and friendly (one short line) and invite them to ask about their documents. Do not fabricate citations for greetings. "
            "Keep formatting simple and readable; use bullet points and short paragraphs where helpful."
        )
        
        prompt = f"{system_prompt}\n\nQuestion: {question}\n\nRelevant context (document and/or web):\n{combined_context}\n\nAnswer clearly and cite sources if they are external."
        
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No response generated."
        
    except Exception as e:
        print(f"[ERROR] Gemini chat error: {e}")
        # Fallback to excerpts
        excerpts = []
        for c in contexts[:3]:
            txt = (c.get('text') or '')[:400]
            excerpts.append(f"- Page {c.get('page')}: {txt}")
        web_excerpts = []
        for i, w in enumerate((web_results or [])[:2], start=1):
            snip = (w.get('snippet') or '')[:300]
            web_excerpts.append(f"- Web {i}: {w.get('title','')} — {snip}\n  URL: {w.get('url','')}")
        
        parts = ["The Gemini API is temporarily unavailable. Here are the most relevant excerpts:"]
        if excerpts:
            parts.append("\nDocument excerpts:\n" + "\n".join(excerpts))
        if web_excerpts:
            parts.append("\nWeb results:\n" + "\n".join(web_excerpts))
        
        return "\n\n".join(parts).strip()

# ----------------------------------------------------------------------------
# Helpers: Mistral chat completion
# ----------------------------------------------------------------------------

def chat_with_mistral(question: str, contexts: List[Dict[str, Any]], web_results: Optional[List[Dict[str, str]]] = None) -> str:
    def build_and_call(model: str, doc_limit: int, doc_slice: int, web_limit: int, web_slice: int) -> requests.Response:
        system_prompt = (
            "You are a helpful assistant that answers questions about the user's document. "
            "Use the provided context passages to answer succinctly. If unsure, say you don't know. "
            "If external web results are provided, you may use them, but clearly indicate sources and do not claim they are in the document. "
            "If the user greets with something like 'hi', 'hello', or 'hey', respond briefly and friendly (one short line) and invite them to ask about their documents. Do not fabricate citations for greetings. "
            "Keep formatting simple and readable; use bullet points and short paragraphs where helpful."
        )
        ctx = contexts[:max(0, doc_limit)]
        context_text = "\n\n".join([f"[Doc Page {c['page']}] {c['text'][:max(0, doc_slice)]}" for c in ctx])
        wr = (web_results or [])[:max(0, web_limit)]
        web_text = "\n\n".join([f"[Web {i+1}] {w.get('title','')} — {w.get('snippet','')[:max(0, web_slice)]}\nURL: {w.get('url','')}" for i, w in enumerate(wr)])
        combined_context = context_text
        if web_text:
            combined_context = (context_text + "\n\nExternal Web Results:\n" + web_text).strip()

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nRelevant context (document and/or web):\n{combined_context}\n\nAnswer clearly and cite sources if they are external."},
        ]
        payload = {"model": model, "messages": messages}
        return requests.post(url, headers=headers, json=payload, timeout=120)

    # Adaptive retry to handle token overflow or similar errors by shrinking context
    attempts = [
        {"doc_limit": 4, "doc_slice": 1200, "web_limit": 3, "web_slice": 400},
        {"doc_limit": 3, "doc_slice": 900,  "web_limit": 2, "web_slice": 300},
        {"doc_limit": 2, "doc_slice": 600,  "web_limit": 1, "web_slice": 200},
        {"doc_limit": 1, "doc_slice": 400,  "web_limit": 0, "web_slice": 0},
    ]

    models_to_try = [MISTRAL_CHAT_MODEL]
    if MISTRAL_FALLBACK_MODEL and MISTRAL_FALLBACK_MODEL != MISTRAL_CHAT_MODEL:
        models_to_try.append(MISTRAL_FALLBACK_MODEL)

    last_err_text = None
    for model in models_to_try:
        for cfg in attempts:
            token_issue = False
            # Retry a few times on capacity errors before shrinking context
            for attempt_idx in range(3):
                r = build_and_call(model, cfg["doc_limit"], cfg["doc_slice"], cfg["web_limit"], cfg["web_slice"])
                if r.status_code < 300:
                    data = r.json()
                    try:
                        return data["choices"][0]["message"]["content"].strip()
                    except Exception:
                        return json.dumps(data)
                else:
                    err = r.text
                    last_err_text = err
                    # Capacity-based retry
                    if 'service_tier_capacity_exceeded' in err or '"code":"3505"' in err:
                        backoff = 0.6 * (2 ** attempt_idx)
                        print(f"[WARN] Mistral service capacity exceeded on model={model}, retrying in {backoff:.1f}s (attempt {attempt_idx+1}/3) with cfg={cfg}")
                        time.sleep(backoff)
                        continue
                    # Token-related: shrink context
                    if "Too many tokens" in err or '"code":"3210"' in err or 'invalid_request_prompt' in err:
                        print(f"[WARN] Mistral token error on model={model}, will shrink context: {cfg}")
                        token_issue = True
                        break
                    # Other errors: try next model configuration
                    print(f"[WARN] Mistral chat error on model={model}: {err}")
                    break
            if token_issue:
                continue
        # Try next model in models_to_try

    # Final extractive fallback: return relevant excerpts so user still gets value
    excerpts = []
    for c in contexts[: min(3, len(contexts))]:
        txt = (c.get('text') or '')[:400]
        excerpts.append(f"- Page {c.get('page')}: {txt}")
    web_excerpts = []
    for i, w in enumerate((web_results or [])[:2], start=1):
        snip = (w.get('snippet') or '')[:300]
        web_excerpts.append(f"- Web {i}: {w.get('title','')} — {snip}\n  URL: {w.get('url','')}")

    parts = [
        "The AI provider is temporarily unavailable. Here are the most relevant excerpts:",
    ]
    if excerpts:
        parts.append("\nDocument excerpts:\n" + "\n".join(excerpts))
    if web_excerpts:
        parts.append("\nWeb results:\n" + "\n".join(web_excerpts))

    return "\n\n".join(parts).strip()

# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="DocuSum Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files for document viewing
app.mount("/files", StaticFiles(directory=str(FILES_DIR), html=False), name="files")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    docs = load_documents()
    return {"documents": docs}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    size = len(contents)
    doc_id = uuid.uuid4().hex
    stored_filename = f"{doc_id}.pdf"
    pdf_path = FILES_DIR / stored_filename
    pdf_path.write_bytes(contents)

    # Extract PDF text per page
    try:
        pages_text = extract_pdf_pages_text(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    # Build FAISS index
    try:
        build_index_for_document(doc_id, pages_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")

    # Persist metadata
    meta = {
        "id": doc_id,
        "name": file.filename,
        "uploadedOn": datetime.utcnow().isoformat(),
        "size": size,
        "type": "pdf",
        "fileUrl": f"/files/{stored_filename}",
    }
    (META_DIR / f"{doc_id}.json").write_text(json.dumps(meta, indent=2))

    docs = load_documents()
    doc = Document(**meta)
    docs.append(doc)
    save_documents(docs)

    return {"document": doc}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    docs = load_documents()
    if not any(d.id == req.docId for d in docs):
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate provider
    if req.provider not in ["mistral", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider must be 'mistral' or 'gemini'")
    
    if req.provider == "gemini" and (not genai or not GEMINI_API_KEY):
        raise HTTPException(status_code=400, detail="Gemini provider is not configured")
    
    if req.provider == "mistral" and not MISTRAL_API_KEY:
        raise HTTPException(status_code=400, detail="Mistral provider is not configured")

    # Retrieve top passages from document
    retrieved = search_doc(req.docId, req.message, top_k=req.max_chunks)
    top_score = max([r["score"] for r in retrieved], default=0.0)

    origin = "document"
    web_sources: Optional[List[WebSource]] = None

    # If we have weak/no doc matches, augment with web search
    if not retrieved or top_score < 0.30:
        web_results = web_search_duckduckgo(req.message, max_results=3)
        if web_results:
            origin = "web" if not retrieved else "hybrid"
            web_sources = [WebSource(title=w["title"], url=w["url"], snippet=w.get("snippet")) for w in web_results]
        else:
            web_results = None
    else:
        web_results = None

    # Compose answer using selected provider
    try:
        if req.provider == "gemini":
            answer = chat_with_gemini(req.message, retrieved, web_results)
        else:  # mistral
            answer = chat_with_mistral(req.message, retrieved, web_results)
    except HTTPException as e:
        # Log and return a graceful fallback message instead of 502 to avoid breaking the UI
        err_detail = getattr(e, "detail", str(e))
        print(f"[ERROR] chat_with_{req.provider} failed: {err_detail}")
        answer = (
            f"I'm sorry, I couldn't generate an answer due to a {req.provider} model error. "
            "Please try again in a moment. If the issue persists, verify the API key and network connectivity."
        )
    except Exception as e:
        # Catch-all to prevent 5xx leaking to client
        print(f"[ERROR] chat_with_{req.provider} unexpected failure: {e}")
        answer = (
            "I'm sorry, I ran into an unexpected error while generating the answer. "
            "Please try again shortly."
        )

    reference: Optional[ChatReference] = None
    if retrieved:
        best = max(retrieved, key=lambda x: x["score"]) if retrieved else None
        if best:
            reference = ChatReference(docId=req.docId, page=int(best["page"]), text=best["text"][:140])

    return {"answer": answer, "reference": reference, "origin": origin, "sources": web_sources}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def web_search_duckduckgo(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                # r has keys: title, href, body
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        print(f"[WARN] DuckDuckGo search failed: {e}")
    return results