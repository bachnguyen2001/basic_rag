import pytest
from app.ai.rag_pipeline import rag_answer

def test_rag_answer():
    query = "I feel anxious about my exams and can't sleep well."
    response = rag_answer(query, top_k=3)
    assert response["query"] == query
    assert len(response["retrieved"]) == 3
    assert isinstance(response["generated"], str)
    assert "disclaimer" in response