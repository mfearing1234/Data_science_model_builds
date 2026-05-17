#!/usr/bin/env python3
"""
Pre-index CSV files into ChromaDB so the chatbot can search them.

Usage:
    python ingest.py                  # index all CSVs in current directory
    python ingest.py path/to/file.csv # index a single file
    python ingest.py path/to/folder/  # index all CSVs in a folder
    python ingest.py --clear          # wipe the index
"""
import sys
import argparse
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "csv_data"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Convert a DataFrame row to a readable text format for embedding
def row_to_text(row: pd.Series, source_name: str) -> str:
    parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
    return f"[{source_name}] " + " | ".join(parts)

# Ingest a single CSV file into the ChromaDB collection
def ingest_csv(csv_path: Path, collection, embed_model: SentenceTransformer) -> int:
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  Skipping {csv_path.name} — empty file")
        return 0

    source = csv_path.name
    texts = [row_to_text(row, source) for _, row in df.iterrows()]
    ids = [f"{source}__{i}" for i in range(len(texts))]

    # Remove any existing entries for this file before re-indexing
    try:
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Embed the texts with a progress bar
    print(f"  Embedding {len(texts)} rows from {source}...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    
    # Add to collection with metadata for source and row number
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=[{"source": source, "row": i} for i in range(len(texts))],
    )
    return len(texts)

# Main function to handle command-line arguments and orchestrate the ingestion process
def main():
    parser = argparse.ArgumentParser(description="Index CSV files for the chatbot")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="CSV file or folder to index (default: current directory)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the entire index and exit",
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if args.clear:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Index cleared.")
        except Exception:
            print("Index was already empty.")
        return
    
    # Get or create the collection for storing CSV data
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Determine which CSV files to index based on the provided path
    target = Path(args.path)
    if target.is_file():
        csv_files = [target] if target.suffix.lower() == ".csv" else []
    else:
        csv_files = sorted(target.rglob("*.csv"))

    # Exit if no CSV files are found
    if not csv_files:
        print(f"No CSV files found in '{target}'")
        sys.exit(1)

    # Load the embedding model once for all files
    print(f"Found {len(csv_files)} CSV file(s). Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    total = 0
    for csv_file in csv_files:
        print(f"\n→ {csv_file}")
        n = ingest_csv(csv_file, collection, embed_model)
        total += n
        print(f"  ✓ {n} rows indexed")

    print(f"\nDone. Total rows in index: {collection.count()}")
    print("Run the chatbot with:\n  streamlit run app.py")


if __name__ == "__main__":
    main()
