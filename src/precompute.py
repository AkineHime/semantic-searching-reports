import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Import the new data loading and cleaning function
from load_data import load_and_clean_data

# Ensure NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')


def normalize_text(text: str) -> str:
    """
    Normalizes text by converting to lowercase, lemmatizing, and removing stopwords.
    This is a model-specific preparation step, separate from the initial cleaning.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Lemmatization and stopword removal
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    
    # Keep words that are not stopwords and have a length > 2
    normalized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return " ".join(normalized_words)

def main():
    """
    Main function to orchestrate the pre-computation pipeline:
    1. Load and clean data
    2. Normalize text
    3. Generate embeddings using a GPU if available
    4. Store results in ChromaDB
    """
    # 1. Load and structurally clean data
    df = load_and_clean_data()

    # 2. Normalize text for the model
    print("Normalizing cleaned text for the model...")
    df['normalized_text'] = [normalize_text(text) for text in tqdm(df['cleaned_text'], desc="Normalizing")]

    # 3. Set up model and device (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading sentence-transformer model...")
    # Using a larger model for better semantic understanding
    model = SentenceTransformer('all-mpnet-base-v2', device=device)

    # 4. Generate embeddings
    print("Generating document embeddings...")
    documents_to_embed = df['normalized_text'].tolist()
    embeddings = model.encode(documents_to_embed, show_progress_bar=True, device=device)

    # 5. Set up ChromaDB and store the data
    print("Setting up ChromaDB...")
    db_path = "./chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    
    # Use a more descriptive collection name
    collection_name = "newsgroups_mpnet"
    
    # Delete collection if it already exists to ensure a fresh start
    if client.get_collection(name=collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it.")
        client.delete_collection(name=collection_name)

    collection = client.create_collection(name=collection_name)

    print(f"Adding {len(df)} documents to ChromaDB collection '{collection_name}'...")
    
    # Prepare data for ChromaDB
    # The document itself will be the cleaned (but not normalized) text for readability.
    # The embeddings are from the normalized text.
    docs_for_db = df['cleaned_text'].tolist()
    metadatas = df[['category', 'filepath']].to_dict('records')
    ids = [str(i) for i in range(len(docs_for_db))]

    # Batch insert for efficiency
    batch_size = 512
    for i in tqdm(range(0, len(docs_for_db), batch_size), desc="Adding to DB"):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=docs_for_db[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print("\\nPre-computation and storage complete.")
    print(f"  - Database path: {db_path}")
    print(f"  - Collection name: {collection_name}")
    print(f"  - Number of documents: {collection.count()}")

if __name__ == '__main__':
    main()
