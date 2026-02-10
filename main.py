from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional
from uuid import uuid4
from dotenv import load_dotenv
import os, json

import numpy as np
from sentence_transformers import SentenceTransformer
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


# MEMORY (USER ONLY)

chat_store: Dict[str, dict] = {}
MAX_MEMORY_TURNS = 8

def trim_memory(memory):
    if len(memory) > MAX_MEMORY_TURNS:
        del memory[:-MAX_MEMORY_TURNS]


# EMBEDDINGS

_model = None
def embedder():
    global _model
    if not _model:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(text):
    return embedder().encode(text, normalize_embeddings=True)

def cosine(a, b):
    return float(np.dot(a, b)) if a.shape == b.shape else -1.0


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

def chunk_text(text, size=250):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def is_doc_question(msg):
    return any(k in msg.lower() for k in ["pdf", "document", "uploaded"])


# PDF UPLOAD

@app.post("/rag/upload")
async def upload_pdf(chat_id: str, file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, chat_id)
    os.makedirs(path, exist_ok=True)

    fname = f"{uuid4()}_{file.filename}"
    full_path = os.path.join(path, fname)

    with open(full_path, "wb") as f:
        f.write(await file.read())

    reader = PdfReader(full_path)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)

    vectors = [{"text": c, "vec": embed(c)} for c in chunk_text(text)]

    chat_store.setdefault(chat_id, {"users": [], "docs": []})
    chat_store[chat_id]["docs"].append(vectors)

    return {
        "file": {
            "name": file.filename,
            "url": f"http://localhost:8000/files/{chat_id}/{fname}"
        }
    }


# CHAT STREAM (REAL, SINGLE SEND)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    chat_id = req.chat_id or str(uuid4())
    chat = chat_store.setdefault(chat_id, {"users": [], "docs": []})

    use_rag = chat["docs"] and is_doc_question(req.message)

    if use_rag:
        q = embed(req.message)
        scored = []
        for d in chat["docs"]:
            for c in d:
                scored.append((c["text"], cosine(q, c["vec"])))
        context = "\n\n".join(t[0] for t in sorted(scored, key=lambda x: x[1], reverse=True)[:2])

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

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )

    final = completion.choices[0].message.content.strip()

    async def stream():
        yield f"data: {json.dumps({'token': final})}\n\n"

        if not use_rag:
            chat["users"].append(req.message)
            trim_memory(chat["users"])

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
