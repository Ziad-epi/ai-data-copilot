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

    class Config:
        env_file = ".env"


settings = Settings()
