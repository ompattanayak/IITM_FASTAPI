from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI()

# Create a ChromaDB client and collection
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
collection = chroma_client.create_collection(name="my_collection", embedding_function=openai_ef)

# Insert sample data if collection is empty
if not collection.get(ids=["1"]).get("ids"):
    collection.add(
        ids=["1", "2"],
        documents=["Bananas are yellow", "Apples are red"],
        metadatas=[{"category": "fruit"}, {"category": "fruit"}]
    )

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search(request: QueryRequest):
    results = collection.query(
        query_texts=[request.query],
        n_results=2
    )
    return {"results": results}

@app.get("/versions")
def versions():
    return {
        "openai_version": openai.__version__,
        "fastapi_version": app.__version__,
        "uvicorn_version": __import__("uvicorn").__version__,
        "chromadb_version": chromadb.__version__,
    }
