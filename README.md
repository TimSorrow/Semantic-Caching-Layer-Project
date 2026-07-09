# Semantic Caching Layer (GCP Hybrid & Local Stack)

A high-performance, context-aware **Semantic Caching Layer** designed for RAG pipelines (such as NotebookLM). It supports **dual-mode deployment**: fully local (FastAPI, Redis VSS, Ollama) and Google Cloud serverless (Cloud Run, Compute Engine, Vertex AI Gemini & Embeddings).

> **Note on the MVP Branch:** This repository contains an experimental `standalone-mvp/` directory which includes a barebones, two-tier semantic cache built with FastAPI, RedisVL, and SentenceTransformers. This serves as a minimal entry point to understand the caching mechanics before diving into the full architecture.

It features a premium, responsive **Glassmorphic Developer Dashboard** styled to match custom portfolio designs.

---

## 🚀 Architectural Features

1. **Semantic Similarity Search:** Evaluates vector embeddings of incoming queries. Semantically identical or rephrased queries are served instantly from the database.
2. **Context-Aware Pre-Filtering (Redis-Native Hybrid Search):**
   Prevents cross-context contamination by restricting vector lookup to specific document boundaries:
   `@context_hash:{context_hash} => [KNN 1 @embedding $vector AS similarity_score]`
3. **Dual AI Engines (Ollama / GCP Vertex AI):**
   - **Local Mode:** Uses Ollama with `llama3.2:latest` (2.0 GB LLM) and `nomic-embed-text` (embeddings).
   - **Cloud Mode:** Serverless integration using Google Vertex AI's `gemini-1.5-flash` and `text-embedding-004` (embeddings) for zero idle VM costs.

---

## ⚡ Premium Developer Dashboard

The project includes an embedded glassmorphic developer console to monitor and test cache performance in real-time.

- **Query Simulator:** A playground to run prompt inputs and similarity thresholds.
- **Pipeline Flow Visualizer:** An animating flow diagram that traces execution paths in real-time with neon connection paths (Green for cache Hits, Purple for Misses).
- **Real-Time Metrics:** Sessions stats including total queries, hit rate, and cumulative latency saved.
- **Live Redis Cache Inspector:** An interactive database inspector to view keys, verify context hashes, delete individual items, or invalidate entire context tags.

*Access the dashboard at `http://127.0.0.1:8000/` after starting the server.*

---

## 📦 Local Setup & Installation

### 1. Configure Ollama Models
Ensure you have [Ollama](https://ollama.com) installed and running. Pull the required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Start Redis Stack
Vector search requires Redis with the RediSearch module. Run it via Docker:
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### 3. Install Dependencies
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root of the project:
```env
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2:latest
SIMILARITY_THRESHOLD=0.90
CACHE_TTL=3600
VECTOR_DIMENSION=768
REDIS_URL=redis://localhost:6379

# GCP Vertex AI settings (optional, defaults to false)
USE_VERTEX_AI=false
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
```

### 5. Run the Server
```bash
python -m uvicorn app.main:app --reload
```

---

## ☁️ Google Cloud Platform (GCP) Deployment

This architecture is optimized to run serverless in GCP, costing virtually **$0/month** by utilising GCP's Free Tier and Vertex AI's pay-as-you-go pricing.

### 1. Host Redis on a Free Compute Engine Instance
1. Spin up an `e2-micro` VM (always free tier under US regions `us-central1`, `us-east1`) running Ubuntu.
2. Install Docker on the VM and launch Redis Stack with a secure password:
   ```bash
   docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 -e REDIS_ARGS="--requirepass YOUR_SECURE_PASSWORD" redis/redis-stack:latest
   ```

### 2. Build and Deploy FastAPI to Cloud Run
1. Authenticate with Google Cloud CLI and set your project:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. Enable the Vertex AI and Artifact Registry APIs:
   ```bash
   gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com
   ```
3. Submit the Docker build:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/semantic-cache-api
   ```
4. Deploy the container to Cloud Run (fully managed serverless):
   ```bash
   gcloud run deploy semantic-cache-api \
     --image gcr.io/YOUR_PROJECT_ID/semantic-cache-api \
     --platform managed \
     --allow-unauthenticated \
     --set-env-vars="REDIS_URL=redis://:YOUR_SECURE_PASSWORD@VM_EXTERNAL_IP:6379,USE_VERTEX_AI=true,GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1"
   ```

---

## 🧪 Running Automated Tests
```bash
python -m pytest
```
