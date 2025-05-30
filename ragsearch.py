from fastapi import FastAPI

app = FastAPI()

@app.get("/versions")
def versions():
    import openai
    import fastapi
    import uvicorn
    import chromadb  # Import here before using
    
    return {
        "openai_version": openai.__version__,
        "fastapi_version": fastapi.__version__,
        "uvicorn_version": uvicorn.__version__,
        "chromadb_version": chromadb.__version__,
    }
