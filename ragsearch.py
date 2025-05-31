import os
import subprocess
import glob
import shutil
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Constants
REPO_URL = "https://github.com/basarat/typescript-book.git"
LOCAL_PATH = "typescript-book"
CHROMA_PATH = ".chroma"
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

# Clone the repo if not already present
if not os.path.exists(LOCAL_PATH):
    subprocess.run(["git", "clone", REPO_URL])

# Optional: Clear Chroma for a clean start
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# FastAPI app setup
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding + Chroma
embedding_fn = OpenAIEmbeddingFunction(api_key=AIPIPE_TOKEN)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="typescript-book", embedding_function=embedding_fn)

# Split large docs into chunks
def chunk_text(text, max_len=1000):
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]

# Load files into Chroma
def load_markdown_files(max_files=100):
    file_paths = glob.glob(f"{LOCAL_PATH}/**/*.md", recursive=True)[:max_files]
    skip_files = {"readme.md", "summary.md"}
    documents, ids = [], []
    doc_id_counter = 0

    for path in file_paths:
        filename = os.path.basename(path).lower()
        if filename in skip_files:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)
                for chunk in chunks:
                    documents.append(chunk[:1000])  # truncate for safety
                    ids.append(f"doc_{doc_id_counter}")
                    doc_id_counter += 1
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if documents:
        collection.add(documents=documents, ids=ids)

# Load on startup if empty
if len(collection.get()["ids"]) == 0:
    load_markdown_files()

# 🔁 OpenRouter Chat Completion via HTTP
def call_openrouter_chat(prompt):
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",  # or your app domain
        "X-Title": "typescript-book-assistant"
    }
    payload = {
        "model": "openai/gpt-4.1-nano",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.post("/add")
async def add_document(request: Request):
    body = await request.json()
    document = body.get("document")
    doc_id = body.get("id", f"doc_{len(collection.get()['ids'])}")
    if not document:
        return JSONResponse(content={"error": "Document is required"}, status_code=400)

    chunks = chunk_text(document)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return {"status": "added", "ids": ids}

@app.post("/search")
async def search_query(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return JSONResponse(content={"error": "Query is required"}, status_code=400)

    results = collection.query(query_texts=[query], n_results=3)
    context_chunks = results.get("documents", [[]])[0]
    context = "\n\n".join(context_chunks)

    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}"
    try:
        answer = call_openrouter_chat(prompt)
    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return {"answer": answer, "context": context}
