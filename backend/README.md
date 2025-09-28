# Mental Health RAG Backend

This is the backend for a Retrieval-Augmented Generation (RAG) application for mental health counseling, built for a Deep Learning course. It uses the `Amod/mental_health_counseling_conversations` dataset (~3.5k Q&A pairs), `sentence-transformers/all-MiniLM-L6-v2` for embeddings, and Google Gemini (`gemini-pro`) for response generation. Vector search is performed in-memory using `faiss` with logging and caching for improved performance.

**Ethical Note**: This is an educational prototype, not a substitute for professional mental health advice. The dataset contains sensitive mental health data; ensure compliance with its RAIL-D license and data privacy regulations (e.g., GDPR, HIPAA).

## Project Structure

- `app/`: Core application code (FastAPI, RAG pipeline).
- `scripts/`: Placeholder for data loading scripts (unused).
- `tests/`: Unit tests for the RAG pipeline.
- `docker/`: Docker configuration for deployment.
- `main.py`: Entry point for the FastAPI application.
- `requirements.txt`: Python dependencies.
- `alembic/`: Unused (can be deleted).

**Parent Directory**: The backend resides in `mental_health_rag/backend/`, with `.env` in `mental_health_rag/`.

## Prerequisites

- Python 3.10+
- Docker (optional, for containerized setup)
- `pip` and virtual environment
- Google Gemini API key (https://ai.google.dev/)

## Setup (Local)

1. **Create `.env` file** in `mental_health_rag/`:
   ```plaintext
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   GENERATION_MODEL=gemini-pro
   API_URL=http://localhost:8000
   TOP_K=3
   GEMINI_API_KEY=your_gemini_api_key_here  # Replace with your Gemini API key


Install dependencies:
cd mental_health_rag/backend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt


Run the server:
uvicorn main:app --reload



Setup (Docker)

Ensure .env exists in mental_health_rag/ with a valid GEMINI_API_KEY.
Run Docker Compose:cd mental_health_rag/backend
docker compose -f docker/docker-compose.yml up --build --remove-orphans



API

Endpoint: POST /api/v1/query
Request:{ "query": "Your question" }


Response:{
  "query": "Your question",
  "retrieved": [["Similar Q", "Response"], ...],
  "generated": "Generated response",
  "disclaimer": "This is not medical advice. Consult a professional for mental health support."
}


Test: curl -X POST http://127.0.0.1:8000/api/v1/query -H "Content-Type: application/json" -d '{"query": "I feel anxious about my exams and can'\''t sleep well."}'



Testing
Run unit tests:
cd mental_health_rag/backend
pytest tests/test_rag.py

Troubleshooting

Gemini API error:
Verify GEMINI_API_KEY in .env is correct.
Check API quota: https://ai.google.dev/.
Check logs in console for error details.


FAISS or embedding issues:
Ensure faiss-cpu and sentence-transformers are installed:pip install faiss-cpu==1.7.4 sentence-transformers==2.2.2




Docker orphan containers warning:
Run with --remove-orphans:docker compose -f docker/docker-compose.yml up --build --remove-orphans




WSL issues:
Ensure Docker Desktop has WSL integration enabled.
Run docker system prune to clear old images/containers if needed.


Slow startup:
Dataset and embeddings are loaded at startup. For larger datasets, consider pre-computing embeddings and saving to disk.



Deep Learning Concepts
This project applies several Deep Learning theories:

Retrieval-Augmented Generation (RAG): Combines retrieval (using faiss for vector search) with generation (Gemini), improving response accuracy for mental health queries (Lewis et al., 2020).
Sentence Embeddings: Uses all-MiniLM-L6-v2, a transformer-based model distilled from BERT, leveraging contrastive learning for semantic similarity.
Transformers: MiniLM relies on self-attention mechanisms for encoding (Vaswani et al., 2017).
Vector Search: faiss enables efficient nearest neighbor search on 384-dimensional embeddings, a practical application of metric learning.

For coursework, you can:

Compare RAG vs. pure Gemini using metrics like ROUGE or BLEU.
Analyze attention weights in MiniLM for mental health keywords (e.g., "anxious").
Fine-tune MiniLM on the dataset for domain-specific performance.

Notes

Dataset: Amod/mental_health_counseling_conversations (~3.5k Q&A pairs).
Optimizations: Includes logging, error handling, and query embedding caching.
Ethical Considerations:
This is a prototype for educational purposes, not for clinical use.
Handle dataset with care due to sensitive mental health content.
Do not deploy publicly without ensuring compliance with data privacy regulations (e.g., GDPR, HIPAA).



Next Steps

Frontend: Develop frontend/ (React/Next.js) to call /api/v1/query.
Speech-to-Text: Integrate functionality for SpeechToText/ (e.g., using Hugging Face whisper).
Fine-tuning: Fine-tune all-MiniLM-L6-v2 for better retrieval.
Scalability: Use faiss.IndexHNSW or pre-compute embeddings for larger datasets.
Deployment: Deploy to AWS/Heroku using Docker.


