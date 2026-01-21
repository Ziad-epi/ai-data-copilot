# ai-data-copilot

Backend foundation for an AI & Big Data product API with RAG retrieval + chat LLM responses.

## Stack
- Python 3.11
- FastAPI
- Uvicorn

## Run with Docker
1) Copy environment file:
   - `cp .env.example .env`
2) Start:
   - `docker compose up --build`
   - Qdrant is exposed on `http://localhost:6333`

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
- `POST /datasets/{dataset_id}/index`
- `POST /datasets/{dataset_id}/search`
- `POST /datasets/{dataset_id}/insights`
- `POST /datasets/{dataset_id}/charts/suggest`
- `POST /datasets/{dataset_id}/report`
- `POST /chat`

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

Index a dataset (RAG v1 retrieval):
```
curl -X POST http://localhost:8000/datasets/<dataset_id>/index \
  -H "Content-Type: application/json" \
  -d '{"columns":["col1","country"],"max_rows":50000,"rows_per_doc":10,"reindex":true}'
```

Search a dataset (returns passages + citations):
```
curl -X POST http://localhost:8000/datasets/<dataset_id>/search \
  -H "Content-Type: application/json" \
  -d '{"query":"top countries","top_k":5,"doc_types":["summary","rows"]}'
```

Compute insights (cached unless force_recompute):
```
curl -X POST http://localhost:8000/datasets/<dataset_id>/insights \
  -H "Content-Type: application/json" \
  -d '{"sample_rows":50000,"target_column":null,"force_recompute":false}'
```

Suggest chart specs:
```
curl -X POST http://localhost:8000/datasets/<dataset_id>/charts/suggest \
  -H "Content-Type: application/json" \
  -d '{"question":"distribution","max_charts":3}'
```

Generate executive report:
```
curl -X POST http://localhost:8000/datasets/<dataset_id>/report
```

Chat with a dataset (RAG first: index before chat):
```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"<dataset_id>","message":"What are the top countries?","top_k":5,"doc_types":["summary","rows"],"response_format":"markdown"}'
```

## Environment Variables
- `STORAGE_DIR`
- `MAX_UPLOAD_MB`
- `PREVIEW_MAX_ROWS`
- `QUERY_MAX_ROWS`
- `SAMPLE_ROWS`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `QDRANT_PATH`
- `RAG_COLLECTION_NAME`
- `RAG_EMBEDDING_MODEL`
- `RAG_ROWS_PER_DOC`
- `RAG_MAX_ROWS_TO_INDEX`
- `RAG_EMBED_BATCH_SIZE`
- `LLM_PROVIDER` (default: `openai_compatible`)
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TEMPERATURE` (default: `0.2`)
- `LLM_MAX_TOKENS` (default: `600`)
- `INSIGHTS_SAMPLE_MAX`
- `INSIGHTS_MISSING_THRESHOLD` (default: `0.3`)
- `INSIGHTS_OUTLIER_METHOD` (default: `iqr` or `zscore`)
- `CHARTS_MAX_POINTS` (default: `50`)

Example LLM config (OpenAI-compatible):
```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=600
```

## Tests
Install dependencies and run:
```
pytest
```
