import os
import glob
import requests
import tiktoken
from dotenv import load_dotenv
from chromadb import Client
from chromadb.config import Settings

load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
MODEL_NAME = "text-embedding-ada-002"

# Load markdown files
def load_documents(path="typescript-book/docs"):
    files = glob.glob(f"{path}/**/*.md", recursive=True)
    docs = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append((file, text))
    return docs

# Split text into chunks (~300 tokens)
def chunk_text(text, chunk_size=300):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [enc.decode(chunk) for chunk in chunks]

# Embed a list of texts
def get_embeddings(texts):
    response = requests.post(
        AIPIPE_EMBEDDING_URL,
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={"input": texts, "model": MODEL_NAME}
    )
    return response.json()["data"]

# Store in ChromaDB
def build_vector_store():
    chroma_client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chromadb"))
    collection = chroma_client.get_or_create_collection("typescript_docs")

    docs = load_documents()
    all_chunks = []
    metadatas = []
    ids = []

    chunk_id = 0
    for file, text in docs:
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        for i, emb in enumerate(embeddings):
            all_chunks.append(chunks[i])
            metadatas.append({"source": file})
            ids.append(f"{chunk_id}")
            chunk_id += 1
            collection.add(documents=[chunks[i]], metadatas=[{"source": file}], embeddings=[emb["embedding"]], ids=[f"{chunk_id}"])

    print(f"Indexed {len(all_chunks)} chunks.")

if __name__ == "__main__":
    build_vector_store()
