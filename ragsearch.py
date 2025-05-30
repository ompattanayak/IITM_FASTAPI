import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import subprocess
import glob

# Constants
REPO_URL = "https://github.com/basarat/typescript-book.git"
LOCAL_PATH = "typescript-book"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Clone the repo if not already cloned
if not os.path.exists(LOCAL_PATH):
    subprocess.run(["git", "clone", REPO_URL])

# Init FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

# New Chroma client setup
chroma_client = chromadb.PersistentClient(path=".chroma")
collection = chroma_client.get_or_create_collection(name="typescript-book", embedding_function=embedding_fn)

# Load markdown files from repo
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

# Only load on first run
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

    # Retrieve context
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n\n".join(results["documents"][0])

    # Construct prompt
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}"

    # Generate answer
    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = completion.choices[0].message.content
    return {"answer": answer, "context": context}
