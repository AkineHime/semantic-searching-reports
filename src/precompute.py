import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

import chromadb
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Import the new data loading and cleaning function
try:
    # When running as a package (python -m src.precompute)
    from src.load_data import load_and_clean_data
except ImportError:
    # When running as a script (python src/precompute.py)
    from load_data import load_and_clean_data

# Ensure NLTK data is available
try:
    stopwords.words("english")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords")
try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download("wordnet")


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, lemmatizing, and removing stopwords."""
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = text.split()
    normalized_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(normalized_words)


def choose_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a sentence transformer model.

    Default model provides a good tradeoff between speed and accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading sentence-transformer model '{model_name}'...")
    return SentenceTransformer(model_name, device=device)


def fit_fuzzy_clusters(
    embeddings: np.ndarray,
    n_components: int,
    random_state: int = 42,
    covariance_type: str = "full",
) -> GaussianMixture:
    """Fit a GaussianMixture (soft clustering) on the embeddings."""
    print(f"Fitting GaussianMixture with n_components={n_components}...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
    )
    gmm.fit(embeddings)
    return gmm


def save_artifacts(
    output_dir: str,
    gmm: GaussianMixture,
    cluster_probs: np.ndarray,
    dominant_cluster: np.ndarray,
):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(gmm, os.path.join(output_dir, "gmm_clustering.joblib"))
    np.save(os.path.join(output_dir, "cluster_probs.npy"), cluster_probs)
    np.save(os.path.join(output_dir, "dominant_cluster.npy"), dominant_cluster)


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings + clusters and store them in ChromaDB.")
    parser.add_argument("--n-clusters", type=int, default=20, help="Number of clusters for soft clustering")
    parser.add_argument("--collection-name", type=str, default="newsgroups_minilm", help="ChromaDB collection name")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to store clustering artifacts")
    args = parser.parse_args()

    # 1. Load and clean data
    df = load_and_clean_data()

    # 2. Normalize text for embedding
    print("Normalizing cleaned text for the model...")
    df["normalized_text"] = [
        normalize_text(text) for text in tqdm(df["cleaned_text"], desc="Normalizing")
    ]

    # 3. Build embeddings
    model = choose_model(args.model_name)
    print("Generating document embeddings...")
    documents_to_embed = df["normalized_text"].tolist()
    embeddings = model.encode(documents_to_embed, show_progress_bar=True, device=model.device)

    # 4. Fit fuzzy clusters
    gmm = fit_fuzzy_clusters(embeddings, n_components=args.n_clusters)
    cluster_probs = gmm.predict_proba(embeddings)
    dominant_cluster = cluster_probs.argmax(axis=1)

    # 5. Persist cluster artifacts
    print(f"Saving clustering artifacts to '{args.artifacts_dir}'...")
    save_artifacts(args.artifacts_dir, gmm, cluster_probs, dominant_cluster)

    # 6. Store embeddings and documents in ChromaDB
    print("Setting up ChromaDB...")
    db_path = "./chroma_db"
    client = chromadb.PersistentClient(path=db_path)

    # Delete collection if it already exists to ensure a clean rebuild
    try:
        existing = client.get_collection(name=args.collection_name)
        if existing is not None:
            print(f"Collection '{args.collection_name}' already exists. Deleting it.")
            client.delete_collection(name=args.collection_name)
    except Exception:
        # If the collection does not exist, `get_collection` may raise NotFoundError.
        # We can safely ignore this and simply create the collection.
        pass

    collection = client.create_collection(name=args.collection_name)

    print(f"Adding {len(df)} documents to ChromaDB collection '{args.collection_name}'...")

    # Attach a small amount of cluster metadata to each document
    metadatas = df[["category", "filepath"]].to_dict("records")
    for i, md in enumerate(metadatas):
        md["dominant_cluster"] = int(dominant_cluster[i])

    docs_for_db = df["cleaned_text"].tolist()
    ids = [str(i) for i in range(len(docs_for_db))]

    batch_size = 512
    for i in tqdm(range(0, len(docs_for_db), batch_size), desc="Adding to DB"):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=docs_for_db[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print("\nPre-computation and storage complete.")
    print(f"  - Database path: {db_path}")
    print(f"  - Collection name: {args.collection_name}")
    print(f"  - Number of documents: {collection.count()}")
    print(f"  - Clustering artifacts: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
