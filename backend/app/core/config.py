from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ai-data-copilot"
    env: str = "dev"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
