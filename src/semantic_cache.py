"""Semantic cache for query embeddings.

This cache is designed to support "semantic" cache hits where two different
queries are close in embedding space. The cache is partitioned by a fast
"dominant cluster" routing value so lookups remain efficient even as the cache
grows.

Cache semantics:
  - Each cache entry stores: query text, normalized query embedding, cached result.
  - Entries are grouped by dominant cluster (integer).
  - Lookup returns the best cached match whose embedding cosine similarity is
    above a configurable threshold.

This is intentionally lightweight and has no external dependencies like Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray  # assumed to be normalized
    result: Any
    inserted_at: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.90,
        max_entries_per_cluster: int = 500,
        max_clusters_to_search: int = 1,
    ):
        """Initialize the semantic cache.

        Args:
            similarity_threshold: cosine similarity threshold for a cache hit.
            max_entries_per_cluster: maximum number of entries to retain in each cluster bucket.
            max_clusters_to_search: how many top clusters to search (by probability) for hits.
        """
        self.similarity_threshold = similarity_threshold
        self.max_entries_per_cluster = max_entries_per_cluster
        self.max_clusters_to_search = max_clusters_to_search

        # cluster_id -> deque[CacheEntry]
        self._buckets: Dict[int, deque[CacheEntry]] = defaultdict(lambda: deque(maxlen=max_entries_per_cluster))

        self.hit_count = 0
        self.miss_count = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-12)

    def lookup(
        self,
        query_embedding: np.ndarray,
        candidate_clusters: List[int],
        query_text: Optional[str] = None,
    ) -> Tuple[bool, Optional[CacheEntry], float]:
        """Attempt to find a cached result.

        Returns:
            (hit, entry, similarity)
        """
        if query_embedding is None:
            return False, None, 0.0

        query_vec = self._normalize(query_embedding)

        best_entry = None
        best_score = 0.0

        # Search for a match in the given cluster buckets (in order)
        for cluster in candidate_clusters[: self.max_clusters_to_search]:
            for entry in self._buckets.get(cluster, []):
                score = self._cosine_similarity(query_vec, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            self.hit_count += 1
            return True, best_entry, best_score

        self.miss_count += 1
        return False, None, best_score

    def add(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: int,
    ) -> None:
        """Add a query + result to the cache under the given dominant cluster."""
        entry = CacheEntry(query=query, embedding=self._normalize(query_embedding), result=result)
        bucket = self._buckets[dominant_cluster]
        bucket.append(entry)

    def stats(self) -> Dict[str, float]:
        total_entries = sum(len(bucket) for bucket in self._buckets.values())
        hit_rate = float(self.hit_count) / max(1, self.hit_count + self.miss_count)
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        self._buckets = defaultdict(lambda: deque(maxlen=self.max_entries_per_cluster))
        self.hit_count = 0
        self.miss_count = 0
