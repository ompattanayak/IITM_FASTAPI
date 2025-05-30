import os
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from git import Repo
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
REPO_URL = "https://github.com/basarat/typescript-book.git"
LOCAL_PATH = "typescript-book"

# Step 1: Clone repo if needed
if not os.path.exists(LOCAL_PATH):
    Repo.clone_from(REPO_URL, LOCAL_PATH)
    print("✅ Cloned TypeScript book repo.")

# Step 2: Load and chunk Markdown files
documents = []
md_files = list(Path(LOCAL_PATH).rglob("*.md"))
for md_file in md_files:
    text = md_file.read_text(encoding="utf-8", errors="ignore")
    chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    for chunk in chunks:
        documents.append({"text": chunk, "source": str(md_file)})

print(f"✅ Loaded {len(documents)} chunks.")

texts = [doc["text"] for doc in documents]
sources = [doc["source"] for doc in documents]

# Step 3: Fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)  # sparse matrix

# Step 4: Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 5: Search endpoint
@app.get("/search")
def search(q: str = Query(..., description="Your search query")):
    query_vec = vectorizer.transform([q])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:3]

    matches = [
        {
            "text": texts[i],
            "source": sources[i],
            "score": float(cosine_similarities[i])
        }
        for i in top_indices if cosine_similarities[i] > 0
    ]

    if not matches:
        return {"answer": "No relevant content found.", "sources": []}

    return {
        "answer": matches[0]["text"],
        "sources": [m["source"] for m in matches]
    }
