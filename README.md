# Semantic Caching Layer (Local & Private RAG Cache)

This project is a fully local, free, and private MVP for a **Semantic Caching Layer** designed for systems like **NotebookLM** or any custom Retrieval-Augmented Generation (RAG) pipelines. The entire stack runs on your local machine and requires zero cloud API keys.

---

## 🚀 Architectural Features

1. **Semantic Similarity Search:** Unlike traditional key-value caches that require exact string matches, this project evaluates the vector embeddings of queries. Rephrased queries with identical semantic intent are served instantly from the cache.
2. **Local Model Stack (Ollama):**
   * **Gemma 4** (`gemma4:latest`) — Used as the local fallback LLM to generate responses on cache misses.
   * **Nomic Embed Text** (`nomic-embed-text`) — Generates 768-dimensional float vectors for input queries.
3. **Context-Aware Pre-Filtering (Redis-Native Hybrid Search):**
   * Cache entries are isolated by a document context state hash (`context_hash`). This prevents outdated responses when the source documents change.
   * Filtering is processed natively inside the Redis vector search engine using **hybrid pre-filtering**:
     `@context_hash:{context_hash} => [KNN 1 @embedding $vector AS similarity_score]`
     This guarantees that Redis searches for similar queries *only* within the correct document context, preventing false cache misses from identical queries in other contexts.

---

## 🛠 Technology Stack

* **FastAPI** — High-performance asynchronous API framework.
* **Redis (Redis Stack)** — Database for vector storage and HNSW index similarity search.
* **Ollama API** — Local inference manager for LLMs and embedding models.
* **HTTPX** — Asynchronous HTTP client to communicate with Ollama.
* **Pytest** — Automated testing suite with mock integrations.

---

## 📦 Quick Start & Setup

### 1. Configure Ollama Models
Ensure you have [Ollama](https://ollama.com) installed and running. Pull the required models:
```bash
ollama pull gemma4
ollama pull nomic-embed-text
```

### 2. Start Redis Stack
Vector search requires Redis with the RediSearch module (included in Redis Stack). Run it via Docker:
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
*The Redis Insight administration panel will be accessible at http://localhost:8001.*

### 3. Install Python Dependencies
Create a virtual environment and install the required packages:
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root of the project (or copy `.env.example`):
```env
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=gemma4:latest
SIMILARITY_THRESHOLD=0.90
CACHE_TTL=3600
VECTOR_DIMENSION=768
REDIS_URL=redis://localhost:6379
```

### 5. Run the FastAPI Server
Start the development server with auto-reload:
```bash
python -m uvicorn app.main:app --reload
```

---

## 🧪 Interactive Testing (Swagger UI)

Open your browser and navigate to: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### Semantic Caching Test Scenario:

1. **Request 1 (Cache MISS):**
   Send a `POST` request to `/api/v1/query` with the following body:
   ```json
   {
     "query": "What is semantic caching?",
     "context_hash": "my_notes_v1"
   }
   ```
   *Response will take ~30-60s as Ollama generates the text using Gemma 4. The response status will be `"status": "MISS"`.*

2. **Request 2 (Cache HIT — Semantically Similar Query):**
   Send a rephrased query within the same context:
   ```json
   {
     "query": "Can you explain semantic caching?",
     "context_hash": "my_notes_v1"
   }
   ```
   *Response will take ~20-50 milliseconds. The response status will be `"status": "HIT"` along with the similarity score (e.g., `"similarity": 0.97`).*

3. **Request 3 (Context Update — Cache MISS):**
   Send the same query but with an updated context version:
   ```json
   {
     "query": "Can you explain semantic caching?",
     "context_hash": "my_notes_v2"
   }
   ```
   *The system detects that the context hash has updated, triggers a `"status": "MISS"`, and generates a fresh response.*

---

## 🧪 Running Automated Tests

To run the unit and integration tests (isolated with mocks):
```bash
python -m pytest
```
