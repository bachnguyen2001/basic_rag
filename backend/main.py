from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.rag import router as rag_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/api/v1/rag")

@app.get("/")
async def root():
    return {"message": "Welcome to the Mental Health Counseling RAG API"}