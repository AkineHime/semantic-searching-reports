import os
import re
import pandas as pd
from tqdm import tqdm

def clean_text(text: str) -> str:
    """
    Cleans a single newsgroup document by removing headers, footers, quotes, and other noise.
    This is a deliberate choice to manually clean the text rather than using built-in library
    functions to have fine-grained control over what is removed, which is crucial for
    generating high-quality semantic embeddings.
    """
    # The header is separated from the body by the first double newline.
    try:
        body_start_index = text.index('\\n\\n')
        text = text[body_start_index + 2:]
    except ValueError:
        # If no double newline, assume the whole text is the body.
        pass

    # Remove lines that are likely quotes
    lines = text.split('\\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('>')]
    text = '\\n'.join(cleaned_lines)

    # Remove signatures starting with '--'
    text = re.sub(r'--.*', '', text, flags=re.DOTALL)

    # Remove email addresses
    text = re.sub(r'\\S*@\\S*\\s?', '', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\\s+', ' ', text).strip()
    
    # The original precompute.py also did lowercasing, stopword removal, and lemmatization.
    # We will keep that logic in the pre-computation step as it's more related to
    # feature engineering for a specific model rather than general data loading.

    return text

def load_and_clean_data(data_path: str = './data/20_newsgroups') -> pd.DataFrame:
    """
    Loads all documents from the 20 Newsgroups directory, cleans them,
    and returns them as a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns ['category', 'filepath', 'raw_text', 'cleaned_text'].
    """
    data = []
    categories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    print(f"Loading and cleaning data from {len(categories)} categories...")

    for category in tqdm(categories, desc="Categories"):
        category_path = os.path.join(data_path, category)
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            try:
                # Use latin1 encoding as it's common for this dataset and avoids errors
                with open(filepath, 'r', encoding='latin1') as f:
                    raw_text = f.read()
                
                cleaned_text = clean_text(raw_text)
                
                # We only care about documents with some content after cleaning
                if cleaned_text:
                    data.append({
                        'category': category,
                        'filepath': filepath,
                        'raw_text': raw_text,
                        'cleaned_text': cleaned_text
                    })
            except Exception as e:
                print(f"Could not read or clean file {filepath}: {e}")

    df = pd.DataFrame(data)
    print(f"Successfully loaded and cleaned {len(df)} documents.")
    return df

if __name__ == '__main__':
    # For testing the script directly
    df = load_and_clean_data()
    print("\\nSample of cleaned data:")
    print(df.head())
    print(f"\\nCleaned text from a sample document ({(df.iloc[0]['filepath'])}):")
    print(df.iloc[0]['cleaned_text'])
