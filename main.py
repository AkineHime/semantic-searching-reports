from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import os
import torch

# Defensive imports with user-friendly errors to help users who forget to install dependencies.
try:
    from fastapi import FastAPI, HTTPException
except ImportError as e:
    raise RuntimeError(
        "Missing dependency 'fastapi'. Install dependencies with: pip install -r requirements.txt"
    ) from e

try:
    from pydantic import BaseModel
except ImportError as e:
    raise RuntimeError(
        "Missing dependency 'pydantic'. Install dependencies with: pip install -r requirements.txt"
    ) from e

try:
    import chromadb
except ImportError as e:
    raise RuntimeError(
        "Missing dependency 'chromadb'. Install dependencies with: pip install -r requirements.txt"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise RuntimeError(
        "Missing dependency 'sentence-transformers'. Install dependencies with: pip install -r requirements.txt"
    ) from e

from src.semantic_cache import SemanticCache


class QueryPayload(BaseModel):
    query: str
    top_n: int = 5


def load_gmm(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GMM clustering artifact not found at '{path}'")
    return joblib.load(path)


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading embedding model '{model_name}'...")
    return SentenceTransformer(model_name, device=device)


def load_vector_store(path: str, collection_name: str):
    client = chromadb.PersistentClient(path=path)
    try:
        return client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load ChromaDB collection '{collection_name}': {e}")


# Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "newsgroups_minilm")
GMM_ARTIFACT_PATH = os.getenv("GMM_ARTIFACT_PATH", "artifacts/gmm_clustering.joblib")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.80"))
CACHE_MAX_ENTRIES_PER_CLUSTER = int(os.getenv("CACHE_MAX_ENTRIES_PER_CLUSTER", "500"))
CACHE_MAX_CLUSTERS_TO_SEARCH = int(os.getenv("CACHE_MAX_CLUSTERS_TO_SEARCH", "1"))


app = FastAPI(title="Semantic Search Service")


# Load resources at startup
try:
    gmm = load_gmm(GMM_ARTIFACT_PATH)
    print(f"Loaded GMM clustering model from '{GMM_ARTIFACT_PATH}'.")
except Exception as e:
    print(f"Warning: Could not load GMM clustering model: {e}")
    gmm = None

try:
    model = load_model(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    model = None

try:
    collection = load_vector_store(CHROMA_DB_PATH, COLLECTION_NAME)
    print(f"Connected to ChromaDB collection '{COLLECTION_NAME}'.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    collection = None

cache = SemanticCache(
    similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
    max_entries_per_cluster=CACHE_MAX_ENTRIES_PER_CLUSTER,
    max_clusters_to_search=CACHE_MAX_CLUSTERS_TO_SEARCH,
)


def _query_embedding(query_text: str) -> np.ndarray:
    if model is None:
        raise RuntimeError("Embedding model is not loaded")
    emb = model.encode([query_text], device=model.device)
    return np.asarray(emb, dtype=np.float32).squeeze(0)


def _dominant_clusters(query_emb: np.ndarray, top_k: int = 1) -> List[int]:
    if gmm is None:
        return [0]
    probs = gmm.predict_proba(query_emb.reshape(1, -1)).ravel()
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [int(x) for x in top_idx]


@app.post("/query")
def query(payload: QueryPayload) -> Dict[str, Any]:
    if collection is None or model is None:
        raise HTTPException(status_code=500, detail="Service not fully initialized")

    query_emb = _query_embedding(payload.query)
    candidate_clusters = _dominant_clusters(query_emb, top_k=CACHE_MAX_CLUSTERS_TO_SEARCH)
    dominant_cluster = candidate_clusters[0] if candidate_clusters else 0

    hit, entry, similarity = cache.lookup(query_emb, candidate_clusters, query_text=payload.query)
    if hit and entry is not None:
        return {
            "query": payload.query,
            "cache_hit": True,
            "matched_query": entry.query,
            "similarity_score": similarity,
            "result": entry.result,
            "dominant_cluster": dominant_cluster,
        }

    # Cache miss: perform vector search
    results = collection.query(
        query_embeddings=query_emb.reshape(1, -1).tolist(),
        n_results=payload.top_n,
    )

    cache.add(
        query=payload.query,
        query_embedding=query_emb,
        result=results,
        dominant_cluster=dominant_cluster,
    )

    return {
        "query": payload.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": similarity,
        "result": results,
        "dominant_cluster": dominant_cluster,
    }


@app.post("/classify")
def classify(payload: QueryPayload) -> Dict[str, Any]:
    """Classify a query by returning cluster probabilities.

    This endpoint is useful for understanding where the query lands in the
    fuzzy cluster space (soft membership).
    """
    if gmm is None or model is None:
        raise HTTPException(status_code=500, detail="Service not fully initialized")

    query_emb = _query_embedding(payload.query)
    probs = gmm.predict_proba(query_emb.reshape(1, -1)).ravel().tolist()
    dominant = int(np.argmax(probs))

    return {
        "query": payload.query,
        "dominant_cluster": dominant,
        "cluster_probs": probs,
    }


@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    return cache.stats()


@app.delete("/cache")
def clear_cache() -> Dict[str, Any]:
    cache.clear()
    return cache.stats()


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to the semantic search service"}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the semantic search API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development mode)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
