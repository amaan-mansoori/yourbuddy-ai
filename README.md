# YourBuddy AI

YourBuddy AI is a FastAPI-based conversational assistant with isolated chat memory and PDF question answering. It uses Groq for generation, Sentence Transformers for embeddings, and retrieval-augmented context from uploaded documents.

## Features

- Streaming chat responses over Server-Sent Events
- Independent conversation memory per `chat_id`
- PDF upload and semantic retrieval
- Personal knowledge and document embedding helpers
- Static access to uploaded files

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Set the Groq API key:

   ```bash
   export GROQ_API_KEY=your_api_key
   ```

3. Start the API:

   ```bash
   uvicorn main:app --reload
   ```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `/docs`.

## Main endpoints

- `POST /chat/stream` - send a message and receive an SSE response
- `POST /rag/upload?chat_id=<id>` - upload a PDF for a chat
- `GET /files/<chat_id>/<filename>` - access an uploaded file

## Notes

Uploaded files and in-memory chat state are local to the running process. Do not commit API keys or sensitive documents.
