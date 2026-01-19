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

## Dataset Endpoints
- `POST /datasets/upload`
- `GET /datasets`
- `GET /datasets/{dataset_id}`

## Example Requests
Upload a CSV:
```
curl -F "file=@data.csv" http://localhost:8000/datasets/upload
```

List datasets:
```
curl http://localhost:8000/datasets
```

Get dataset metadata:
```
curl http://localhost:8000/datasets/<dataset_id>
```

## Tests
Install dependencies and run:
```
pytest
```
