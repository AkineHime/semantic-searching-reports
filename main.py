from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# 1. Initialize ChromaDB client and get collection
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="newsgroups_mpnet")
    print("Successfully connected to ChromaDB collection 'newsgroups_mpnet'.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    collection = None

# 2. Load the sentence-transformer model
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    print("Successfully loaded sentence-transformer model.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class SearchQuery(BaseModel):
    query: str
    top_n: int = 5

@app.post("/search/")
def search(query: SearchQuery):
    if not collection or not model:
        return {"error": "Database or model not initialized correctly. Please check server logs."}

    # 4. Generate an embedding for the query
    query_embedding = model.encode([query.query], device=device)

    # 5. Query the ChromaDB collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=query.top_n
    )

    return results

@app.get("/")
def read_root():
    return {"message": "Welcome to the 20 Newsgroups Similarity Search API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
