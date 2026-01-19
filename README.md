# ai-data-copilot

Backend foundation for an AI & Big Data product API. No AI or RAG at this stage.

## Stack
- Python 3.11
- FastAPI
- Uvicorn

## Run with Docker
1) Copy environment file:
   - `cp .env.example .env`
2) Start:
   - `docker compose up --build`

## Health & Docs
- Health: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`
