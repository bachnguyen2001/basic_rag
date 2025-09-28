from fastapi import FastAPI
from app.api.v1 import rag

app = FastAPI(title="Mental Health RAG API")
app.include_router(rag.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Mental Health RAG API"}