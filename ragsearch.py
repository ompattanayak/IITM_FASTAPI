import os
import subprocess
import glob
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import openai

# Constants
REPO_URL = "https://github.com/basarat/typescript-book.git"
LOCAL_PATH = "typescript-book"
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

# Configure OpenAI client
openai.api_key = AIPIPE_TOKEN
openai.api_base = "https://aipipe.org/openrouter/v1"

# Clone the repo if not already present
if not os.path.exists(LOCAL_PATH):
    subprocess.run(["git", "clone", REPO_URL])

# FastAPI app init
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding function & Chroma client
embedding_fn = OpenAIEmbeddingFunction(api_key=AIPIPE_TOKEN)
chroma_client = chromadb.PersistentClient(path=".chroma")
collection = chroma_client.get_or_create_collection(name="typescript-book", embedding_function=embedding_fn)

# Load markdown files as docs
def load_markdown_files():
    file_paths = glob.glob(f"{LOCAL_PATH}/**/*.md", recursive=True)
    documents, ids = [], []
    for i, path in enumerate(file_paths):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)
                ids.append(f"doc_{i}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
    if documents:
        collection.add(documents=documents, ids=ids)

# Load once if not already loaded
if len(collection.get()["ids"]) == 0:
    load_markdown_files()

@app.post("/add")
async def add_document(request: Request):
    body = await request.json()
    document = body.get("document")
    doc_id = body.get("id", f"doc_{len(collection.get()['ids'])}")
    if not document:
        return JSONResponse(content={"error": "Document is required"}, status_code=400)
    collection.add(documents=[document], ids=[doc_id])
    return {"status": "added", "id": doc_id}

@app.post("/search")
async def search_query(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return JSONResponse(content={"error": "Query is required"}, status_code=400)

    # Retrieve top 3 docs
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n\n".join(results["documents"][0])

    # Build prompt
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}"

    # Generate response from AIPipe model
    response = openai.ChatCompletion.create(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = response['choices'][0]['message']['content']
    return {"answer": answer, "context": context}
