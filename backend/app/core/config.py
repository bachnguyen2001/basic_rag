from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EMBEDDING_MODEL: str
    GENERATION_MODEL: str
    API_URL: str
    TOP_K: int
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
