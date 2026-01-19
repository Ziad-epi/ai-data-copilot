import logging

from fastapi import FastAPI

from app.api import router as api_router
from app.core.config import settings


def create_app() -> FastAPI:
    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger("ai-data-copilot")
    logger.info("Starting %s (%s)", settings.app_name, settings.env)

    app = FastAPI(title=settings.app_name)
    app.include_router(api_router)
    return app


app = create_app()
