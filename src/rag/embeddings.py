"""
Embedding model management functions.
This module provides pure functions for creating and using HuggingFace embedding models.
All functions are stateless and composable.
"""

from langchain_huggingface import HuggingFaceEmbeddings


# =======================================
# 1. Establish a embedding model
# =======================================
def create_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
) -> HuggingFaceEmbeddings:
    """
    Create a HuggingFace embedding model.

    Args:
        model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# =======================================
# 2. Embedding the Document object
# =======================================
def embed_texts(
    embedding_model: HuggingFaceEmbeddings, texts: list[str]
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (batch processing).

    Args:
        embedding_model: HuggingFaceEmbeddings instance
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    return embedding_model.embed_documents(texts)


# =======================================
# 3. Embedding the user query
# =======================================
def embed_query(embedding_model: HuggingFaceEmbeddings, query: str) -> list[float]:
    """
    Generate embedding for a single query text.

    Args:
        embedding_model: HuggingFaceEmbeddings instance
        query: Query text to embed

    Returns:
        Embedding vector
    """
    return embedding_model.embed_query(query)
