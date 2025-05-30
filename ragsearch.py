from fastapi import FastAPI

app = FastAPI()

@app.get("/versions")
def versions():
    import openai
    import fastapi
    import uvicorn
    return {
        "openai_version": openai.__version__,
        "fastapi_version": fastapi.__version__,
        "uvicorn_version": uvicorn.__version__,
    }
