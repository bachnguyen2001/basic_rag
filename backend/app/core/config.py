from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3] 
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    EMBEDDING_MODEL: str
    GENERATION_MODEL: str
    API_URL: str
    TOP_K: int
    GEMINI_API_KEY: str

    class Config:
        env_file = ENV_PATH
        env_file_encoding = "utf-8"

settings = Settings()
