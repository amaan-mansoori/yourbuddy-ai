from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional
from uuid import uuid4
from dotenv import load_dotenv
import os, json, time

import numpy as np
from sentence_transformers import SentenceTransformer
import groq
from groq import Groq
from pypdf import PdfReader

# ENV
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# APP

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# MEMORY (USER ONLY)

chat_store: Dict[str, dict] = {}
MAX_MEMORY_TURNS = 8
CHAT_EXPIRY_SECONDS = 3600  # 1 hour


def trim_memory(memory):
    if len(memory) > MAX_MEMORY_TURNS:
        memory[:] = memory[-MAX_MEMORY_TURNS:]


def cleanup_old_chats():
    """Remove chat sessions that have been idle for longer than CHAT_EXPIRY_SECONDS."""
    now = time.time()
    expired = [
        cid for cid, data in list(chat_store.items())
        if now - data.get("last_active", now) > CHAT_EXPIRY_SECONDS
    ]
    for cid in expired:
        del chat_store[cid]


# EMBEDDINGS

_model = None
def embedder():
    global _model
    if not _model:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(text):
    return embedder().encode(text, normalize_embeddings=True)


def embed_batch(texts):
    """Encode a list of texts in a single batched forward pass."""
    return embedder().encode(texts, normalize_embeddings=True, batch_size=64)


# MODELS

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None


# SYSTEM PROMPTS

BASE_SYSTEM = {
    "role": "system",
    "content": (
        "You are **YourBuddy AI**, a ChatGPT-like assistant built by Amaan Mansoori.\n\n"

        "CRITICAL IDENTITY RULES\n\n"
        "- You are built by Amaan Mansoori\n"
        "- You are NOT built by OpenAI\n"
        "- NEVER mention OpenAI, GPT, or ChatGPT\n"
        "- NEVER change your creator\n"
        "- If asked who built you, say exactly:\n"
        "  'YourBuddy AI was built by Amaan Mansoori.'\n\n"

        "ABOUT THE CREATOR\n\n"
        "If asked 'Who is Amaan Mansoori?', respond briefly and professionally:\n\n"
        "- He lives in Vidisha, India\n"
        "- He has completed B.Tech from LNCT\n"
        "- He is an aspiring Software Development Engineer (SDE)\n"
        "- He is interested in technology, programming, and AI systems\n"
        "- Do NOT exaggerate or invent achievements\n\n"

        "CORE BEHAVIOR RULES\n\n"
        "- Be accurate, honest, and structured\n"
        "- NEVER hallucinate or guess meanings\n"
        "- Do NOT make assumptions unless explicitly stated\n"
        "- Use headings, spacing, and formatting like ChatGPT\n"
        "- Preserve numbering and line breaks exactly\n"
        "- NEVER merge numbered lines into one sentence\n\n"

        "CONVERSATION & CONTEXT RULES\n\n"
        "- Maintain context ONLY within the same chat\n"
        "- This is a continuous conversation inside one chat\n"
        "- Follow-up questions refer to previous messages\n"
        "- A new chat always means fresh context\n"
        "- NEVER leak or reuse context from other chats\n\n"

        "INTELLIGENCE RULES\n\n"
        "- Always understand the user's intent before answering\n"
        "- 'RAG agents' always means:\n"
        "  Retrieval-Augmented Generation agents (AI context)\n"
        "- It NEVER refers to rag collectors or unrelated meanings\n\n"

        "FACT MODE\n\n"
        "- Answer factual questions only\n"
        "- Be precise and to the point\n"
        "- No storytelling or opinions\n\n"

        "TECH MODE\n\n"
        "- Act as a senior technical assistant\n"
        "- Explain concepts clearly and logically\n"
        "- Use correct and standard terminology\n"
        "- Never invent definitions\n\n"

        "COUNTING / LISTING / PATTERN RULES\n\n"
        "- If the user asks for counting, steps, lists, or patterns:\n"
        "- ALWAYS print the full output\n"
        "- Each item must be on a new line\n"
        "- Lists must be vertical, never single-line\n"
        "- Never summarize or skip values\n\n"

        "FORMATTING RULES (VERY IMPORTANT)\n\n"
        "- Use clean spacing and readable paragraphs\n"
        "- Headings must be on their own line\n"
        "- Use bullet points only when they add clarity\n"
        "- Counting and sequences must be line-by-line\n"
        "- Code MUST be inside fenced code blocks\n"
        "- Always specify the programming language\n\n"

        "CHAT ISOLATION RULES\n\n"
        "- Each chat_id is an independent conversation\n"
        "- NEVER reuse old context incorrectly\n"
    )
}

RAG_SYSTEM = {
    "role": "system",
    "content": (
        "Answer strictly from document context.\n"
        "No repetition.\n"
        "If not found reply exactly:\n"
        "\"I don't have this information in the uploaded document.\""
    )
}


# HELPERS

def chunk_text(text, size=250, overlap=50):
    """Split text into overlapping word-based chunks for better context retrieval."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i + size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

def is_doc_question(msg):
    return any(k in msg.lower() for k in ["pdf", "document", "uploaded"])


# PDF UPLOAD

@app.post("/rag/upload")
async def upload_pdf(chat_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    path = os.path.join(UPLOAD_DIR, chat_id)
    os.makedirs(path, exist_ok=True)

    fname = f"{uuid4()}_{file.filename}"
    full_path = os.path.join(path, fname)

    with open(full_path, "wb") as f:
        f.write(content)

    reader = PdfReader(full_path)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)

    # Batch-encode all chunks in a single model pass for efficiency
    chunks = chunk_text(text)
    chunk_vecs = embed_batch(chunks)
    vectors = [{"text": c, "vec": v} for c, v in zip(chunks, chunk_vecs)]

    chat = chat_store.setdefault(chat_id, {"users": [], "docs": [], "last_active": time.time()})
    chat["docs"].append(vectors)
    chat["last_active"] = time.time()

    return {
        "file": {
            "name": file.filename,
            "url": f"{BASE_URL}/files/{chat_id}/{fname}"
        }
    }


# CHAT STREAM (REAL, SINGLE SEND)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    cleanup_old_chats()

    chat_id = req.chat_id or str(uuid4())
    chat = chat_store.setdefault(chat_id, {"users": [], "docs": [], "last_active": time.time()})
    chat["last_active"] = time.time()

    use_rag = chat["docs"] and is_doc_question(req.message)

    if use_rag:
        q = embed(req.message)
        # Flatten all chunks and run vectorized similarity in one matrix multiply
        all_chunks = [c for d in chat["docs"] for c in d]
        if all_chunks:
            vecs = np.stack([c["vec"] for c in all_chunks])
            scores = vecs @ q  # dot products for all vectors at once (L2-normalised)
            top_indices = np.argsort(scores)[::-1][:2]
            context = "\n\n".join(all_chunks[i]["text"] for i in top_indices)
        else:
            context = ""

        messages = [
            RAG_SYSTEM,
            {"role": "system", "content": context},
            {"role": "user", "content": req.message}
        ]
    else:
        messages = [BASE_SYSTEM]
        for u in chat["users"]:
            messages.append({"role": "user", "content": u})
        messages.append({"role": "user", "content": req.message})

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        final = completion.choices[0].message.content.strip()
    except groq.APIError:
        async def err_stream():
            yield f"data: {json.dumps({'token': 'Sorry, I encountered an error. Please try again.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    async def stream():
        yield f"data: {json.dumps({'token': final})}\n\n"

        if not use_rag:
            chat["users"].append(req.message)
            trim_memory(chat["users"])

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
