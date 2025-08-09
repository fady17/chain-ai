# ===== app/config.py =====
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "RAG API Service"
    upload_dir: Path = Path("uploads")
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: set = {".pdf", ".txt", ".md", ".py"}
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 3
    debug: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()

