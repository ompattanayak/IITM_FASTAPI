from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/search")
def search(q: str):
    return JSONResponse(content={"answer": f"You searched for: {q}"})



# import os
# from pathlib import Path
# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# from git import Repo
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings
# import os

# # Constants
# REPO_URL = "https://github.com/basarat/typescript-book.git"
# LOCAL_PATH = "typescript-book"
# CHROMA_COLLECTION_NAME = "typescript-book"

# # Step 1: Clone the GitHub repo if not already present
# if not os.path.exists(LOCAL_PATH):
#     Repo.clone_from(REPO_URL, LOCAL_PATH)
#     print("✅ Cloned TypeScript book repo.")

# # Step 2: Load and chunk Markdown files
# documents = []
# md_files = list(Path(LOCAL_PATH).rglob("*.md"))
# for md_file in md_files:
#     text = md_file.read_text(encoding="utf-8", errors="ignore")
#     chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
#     for chunk in chunks:
#         documents.append({"text": chunk, "source": str(md_file)})

# print(f"✅ Loaded {len(documents)} chunks.")

# # Step 3: Setup embedding model
# model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
# texts = [doc["text"] for doc in documents]
# metadatas = [{"source": doc["source"]} for doc in documents]
# embeddings = model.encode(texts).tolist()

# # Step 4: Setup ChromaDB
# chroma_client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory=".chroma"
# ))
# collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# # Only populate collection if empty
# if collection.count() == 0:
#     collection.add(
#         documents=texts,
#         metadatas=metadatas,
#         ids=[str(i) for i in range(len(texts))],
#         embeddings=embeddings
#     )
#     print("✅ ChromaDB populated.")
# else:
#     print("ℹ️ ChromaDB already has data.")

# # Step 5: Create FastAPI app
# app = FastAPI()

# # Enable CORS for all origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Step 6: Search endpoint
# @app.get("/search")
# def search(q: str = Query(..., description="Your search query")):
#     query_embedding = model.encode([q])[0].tolist()
#     results = collection.query(query_embeddings=[query_embedding], n_results=3)

#     matches = [
#         {
#             "text": doc,
#             "source": meta["source"]
#         }
#         for doc, meta in zip(results["documents"][0], results["metadatas"][0])
#     ]

#     return {
#         "answer": matches[0]["text"] if matches else "No relevant content found.",
#         "sources": [m["source"] for m in matches]
#     }
