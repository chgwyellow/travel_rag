"""
Setup ChromaDB vector database.
This script builds the vector database from processed documents:
1. Load documents from processed data
2. Create embedding model
3. Create ChromaDB vector store
4. Generate and store embeddings
5. Verify results
"""

import json

from src.config import CHROMA_DB_DIR, PROCESSED_DATA_DIR
from src.rag.embeddings import create_embedding_model
from src.rag.vector_store import (
    add_documents_to_store,
    create_chroma_client,
    create_vector_store,
    get_collection_count,
    get_or_create_collection,
)
from src.utils.emoji_log import done, error, success, task


# =======================================
# Helper Functions
# =======================================
def load_documents(city_name: str) -> list[dict]:
    """Load documents from JSON file."""
    try:
        file_path = PROCESSED_DATA_DIR / f"{city_name}_attractions_documents.json"

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            done(f"{city_name} attraction documents has retrieved")
            return data
    except FileNotFoundError:
        error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        error(f"Invalid JSON in {city_name} documents: {e}")
        raise


def prepare_data(documents_data: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    Prepare documents, metadatas, and ids.

    Args:
        documents_data: List of document dicts from JSON

    Returns:
        Tuple of (documents, metadatas, ids)
        - documents: List of document texts
        - metadatas: List of metadata dicts
        - ids: List of unique IDs
    """
    documents = []
    metadatas = []
    ids = []

    for doc in documents_data:
        # Retrieve document text
        documents.append(doc["document"])

        # Retrieve metadata
        metadatas.append(doc["metadata"])

        # Retrieve ID
        ids.append(doc["place_id"])

    done("Needed data has been prepared")

    return documents, metadatas, ids


# =======================================
# Main Function
# =======================================
def setup_vector_database(
    city_name: str = "Seattle",
    collection_name: str = "travel_attractions",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Setup ChromaDB vector database for a city.

    Args:
        city_name: Name of the city
        collection_name: ChromaDB collection name
        embedding_model_name: HuggingFace model name
    """
    task("Start building vector database and writing documents...")

    # 1. Retrieve the attraction data
    city_raw_data = load_documents(city_name=city_name)

    # 2. Prepare data
    documents, metadatas, ids = prepare_data(documents_data=city_raw_data)

    # 3. Embedding model
    model = create_embedding_model()
    done("Embedding model built.")

    # 4. Create chromadb client and create or get collection
    client = create_chroma_client()
    done("client built")

    collection = get_or_create_collection(
        client=client, collection_name=collection_name
    )
    done("Collection has been created or retrieved")

    # 5. vector store
    vector_store = create_vector_store(
        collection_name=collection_name, embeddings=model
    )
    done("Vector store has been built")

    # 6. Embedding documents
    add_documents_to_store(
        vector_store=vector_store, documents=documents, metadatas=metadatas, ids=ids
    )
    done("Documents has been written to ChromaDB")

    # 7. Verify
    collection_count = get_collection_count(collection=collection)

    success(
        f"Embedding {collection_count} documents of {city_name} attraction to {CHROMA_DB_DIR}"
    )


# =======================================
# CLI Entry Point
# =======================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup ChromDB vector database")
    parser.add_argument("--city", default="Seattle", help="City name")
    parser.add_argument(
        "--collection", default="travel_attractions", help="Collection name"
    )

    args = parser.parse_args()

    setup_vector_database(city_name=args.city, collection_name=args.collection)
