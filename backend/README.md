# Mental Health RAG Backend

This is the backend for a Retrieval-Augmented Generation (RAG) application for mental health counseling, built for a Deep Learning course. It uses the `Amod/mental_health_counseling_conversations` dataset (~3.5k Q&A pairs), `sentence-transformers/all-MiniLM-L6-v2` for embeddings, and GPT-2 for response generation. Data is stored in PostgreSQL with the `pgvector` extension for vector search (384-dimensional embeddings).

**Ethical Note**: This is an educational prototype, not a substitute for professional mental health advice. The dataset contains sensitive mental health data; ensure compliance with its RAIL-D license and data privacy regulations (e.g., GDPR, HIPAA).

## Project Structure

- `app/`: Core application code (FastAPI, SQLAlchemy, RAG pipeline).
- `scripts/`: Script to load dataset into PostgreSQL.
- `tests/`: Unit tests for the RAG pipeline.
- `docker/`: Docker configuration for deployment.
- `main.py`: Entry point for the FastAPI application.
- `requirements.txt`: Python dependencies.
- `alembic/`: Unused (can be deleted, as schema is created directly via SQLAlchemy).

**Parent Directory**: The backend resides in `mental_health_rag/backend/`, with `.env` and `init-db.sql` in `mental_health_rag/`.

## Prerequisites

- Python 3.10+
- PostgreSQL 15+ with `pgvector` extension
- Docker (optional, for containerized setup)
- `pip` and virtual environment

## Setup (Local)

1. **Create `.env` file** in `mental_health_rag/` (project root):
   ```plaintext
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   GENERATION_MODEL=gpt2
   API_URL=http://localhost:8000
   TOP_K=3
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=rag_ai
   DB_USER=admin
   DB_PASSWORD=admin123