"""
ChromaDB vector store management functions.
This module provides pure functions for creating and managing ChromaDB vector stores.
All
"""

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DB_DIR


# =======================================
# 1. Create ChromaDB client
# =======================================
def create_chroma_client(
    persist_directory: str = CHROMA_DB_DIR,
) -> chromadb.PersistentClient:
    """
    Create a ChromaDB persistent client.

    Args:
        persist_directory: Directory to store ChromaDB data

    Returns:
        ChromaDB PersistentClient instance
    """
    return chromadb.PersistentClient(path=persist_directory)


# =======================================
# 2. Get or create collection
# =======================================
def get_or_create_collection(
    client: chromadb.PersistentClient, collection_name: str, metadata: dict = None
) -> chromadb.Collection:
    """
    Get existing collection or create new one.

    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        metadata: Optional metadata for the collection

    Returns:
        ChromaDB Collection instance
    """
    if metadata is None:
        metadata = {"description": "Vector database collection"}

    return client.get_or_create_collection(name=collection_name, metadata=metadata)


# =======================================
# 3. Create vector store
# =======================================
def create_vector_store(
    collection_name: str,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    """
    Create a LangChain Chroma vector store.

    Args:
        collection_name: Name of the collection
        embeddings: HuggingFace embeddings instance
        persist_directory: Directory to store ChromaDB data

    Returns:
        LangChain Chroma instance
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


# =======================================
# 4. Add documents to vector store
# =======================================
def add_documents_to_store(
    vector_store: Chroma, documents: list[str], metadatas: list[dict], ids: list[str]
) -> None:
    """
    Add documents to vector store with batch processing.

    Args:
        vector_store: LangChain Chroma instance
        documents: List of document texts
        metadatas: List of metadata dicts
        ids: List of unique IDs

    Returns:
        None
    """
    vector_store.add_texts(texts=documents, metadatas=metadatas, ids=ids)
