from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ai-data-copilot"
    env: str = "dev"
    log_level: str = "INFO"
    storage_dir: str = "/app/storage"
    max_upload_mb: int = 20
    preview_max_rows: int = 200
    query_max_rows: int = 1000
    sample_rows: int = 50000
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_path: str | None = None
    rag_collection_name: str = "datasets"
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_rows_per_doc: int = 10
    rag_max_rows_to_index: int = 50000
    rag_embed_batch_size: int = 64

    class Config:
        env_file = ".env"


settings = Settings()
