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
- `GET /datasets/{dataset_id}/preview`
- `GET /datasets/{dataset_id}/schema`
- `POST /datasets/{dataset_id}/query`

## Example Requests
Upload a CSV:
```
curl -F "file=@data.csv" http://localhost:8000/datasets/upload
```

Upload with delimiter:
```
curl -F "file=@data.csv" "http://localhost:8000/datasets/upload?delimiter=;"
```

List datasets:
```
curl http://localhost:8000/datasets
```

Get dataset metadata:
```
curl http://localhost:8000/datasets/<dataset_id>
```

Preview rows:
```
curl "http://localhost:8000/datasets/<dataset_id>/preview?limit=20"
```

Get schema and stats:
```
curl http://localhost:8000/datasets/<dataset_id>/schema
```

Query with filters:
```
curl -X POST http://localhost:8000/datasets/<dataset_id>/query \
  -H "Content-Type: application/json" \
  -d '{"columns":["col1","country"],"filters":{"country":"FR"},"limit":10}'
```

## Environment Variables
- `STORAGE_DIR`
- `MAX_UPLOAD_MB`
- `PREVIEW_MAX_ROWS`
- `QUERY_MAX_ROWS`
- `SAMPLE_ROWS`

## Tests
Install dependencies and run:
```
pytest
```
