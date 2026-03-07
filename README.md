# Semantic Search Service

A **semantic search** service built on the 20 Newsgroups dataset.

It provides:
- **Corpus cleaning + embedding** using `sentence-transformers`
- **Fuzzy clustering** via `GaussianMixture` (soft cluster probabilities)
- **Semantic cache** (cache hits triggered by embedding similarity, not identical text)
- **FastAPI service** with endpoints for query + cache metrics

---

## 🚀 Quickstart (Windows PowerShell)

### 1) Clone the repo

```powershell
git clone <your-repo-url> semantic-search
cd semantic-search
```

### 2) Create + activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 📦 Download the dataset (not included in the repo)

This project expects the 20 Newsgroups data to live in `data/20_newsgroups/`.

### ✅ Recommended (one step)

Run the included helper script. It downloads the official UCI archive and extracts it into the expected layout:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

```powershell
python download_20newsgroups.py
```

> ⚠️ This dataset is intentionally **not** checked into git (it is large).

---

## 🔥 Precompute embeddings + clustering

Once the dataset is downloaded and extracted into `data/20_newsgroups/`, run the preprocessing pipeline to build the vector store and clustering artifacts:

```powershell
python src/precompute.py --n-clusters 20 --collection-name newsgroups_minilm
```

Artifacts produced:
- `chroma_db/` (vector store)
- `artifacts/gmm_clustering.joblib` (GMM cluster model)
- `artifacts/cluster_probs.npy` (soft membership probabilities)
- `artifacts/dominant_cluster.npy` (dominant cluster per document)

---

## 🧠 Run the API

```powershell
python main.py --host 0.0.0.0 --port 8000 --reload
```

### Available endpoints

- `POST /query` — run a semantic query (uses the cache)
- `POST /classify` — returns cluster probabilities for a query
- `GET /cache/stats` — cache hit/miss metrics
- `DELETE /cache` — clear the cache

---

## 🧪 Example request

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method POST -ContentType 'application/json' -Body '{"query":"what is machine learning"}'
```

Response includes:
- `cache_hit` (bool)
- `matched_query` (cached query that matched)
- `similarity_score` (cosine similarity)
- `result` (vector search results)
- `dominant_cluster` (cluster used for routing)

---

## 🧩 Notes

- The cache is **in-memory** and resets when the service restarts.
- Cache similarity threshold is configurable via `CACHE_SIMILARITY_THRESHOLD` (env var).
- For best performance, use a GPU or a smaller model.
