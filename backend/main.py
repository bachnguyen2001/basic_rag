from fastapi import FastAPI
from app.api.v1 import rag
from app.core.database import Base, engine
from app.ai.embeddings import load_and_index_data

app = FastAPI(
    title="Mental Health RAG API",
    description="RAG API for mental health counseling. Not for medical advice.",
)

# Tạo DB tables và index data lúc startup
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)  # Tạo tables
    load_and_index_data()  # Load dataset vào DB và tạo embeddings

app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
