from fastapi import APIRouter
from pydantic import BaseModel
from ...ai.rag_pipeline import rag_answer

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
async def query_rag(request: QueryRequest):
    return await rag_answer(request.query)