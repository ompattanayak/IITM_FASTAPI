
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class InputData(BaseModel):
    docs: List[str]
    query: str

@app.post("/similarity")
def get_similarity(data: InputData):
    vectorizer = TfidfVectorizer()
    all_text = data.docs + [data.query]
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    results = sorted(zip(data.docs, scores), key=lambda x: x[1], reverse=True)

    return {
        "matches": [{"doc": doc, "score": float(score)} for doc, score in results]
    }

# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from typing import List
# from sentence_transformers import SentenceTransformer, util

# app = FastAPI()

# model = SentenceTransformer('all-MiniLM-L6-v2')

# class InputData(BaseModel):
#     docs: List[str]
#     query: str

# @app.post("/similarity")
# def get_similarity(data: InputData):
#     doc_embeddings = model.encode(data.docs, convert_to_tensor=True)
#     query_embedding = model.encode(data.query, convert_to_tensor=True)

#     cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
#     results = sorted(zip(data.docs, cosine_scores), key=lambda x: x[1], reverse=True)

#     top_matches = [{"doc": doc, "score": float(score)} for doc, score in results]
#     return {"matches": top_matches}

#=================================
# # main.py
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import chromadb
# from chromadb.utils import embedding_functions
# import asyncio

# # Request body model
# class SimilarityRequest(BaseModel):
#     docs: List[str]
#     query: str

# # Response model
# class SimilarityResponse(BaseModel):
#     matches: List[str]

# app = FastAPI()

# # Enable CORS - allow all origins, headers, and POST, OPTIONS methods
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_methods=["POST", "OPTIONS"],
#     allow_headers=["*"],
# )

# # Initialize ChromaDB client and collection globally
# client = chromadb.PersistentClient(path="./vector_db")
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="BAAI/bge-base-en-v1.5"
# )

# # Try to get existing collection, or create new
# try:
#     collection = client.get_collection(name="documents")
# except Exception:
#     collection = client.create_collection(
#         name="documents",
#         embedding_function=embedding_function
#     )


# @app.post("/similarity", response_model=SimilarityResponse)
# async def similarity_search(body: SimilarityRequest):
#     # return {"matches": ["Tesla builds electric vehicles"]}
#     docs = body.docs
#     query = body.query

#     if not docs:
#         raise HTTPException(status_code=400, detail="Docs list cannot be empty.")
#     if not query:
#         raise HTTPException(status_code=400, detail="Query string cannot be empty.")

#     # Clear the collection and add new docs for this request
#     # (assuming ephemeral data per request, adjust as needed)
#     collection.delete(where={})  # deletes all entries
#     ids = [str(i) for i in range(len(docs))]
#     collection.add(documents=docs, ids=ids)

#     # Query top 3 most similar docs
#     results = collection.query(query_texts=[query], n_results=3)

#     matched_docs = results["documents"][0]  # top 3 matched document texts

#     return SimilarityResponse(matches=matched_docs)
