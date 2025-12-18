"""
Citation formatting for RAG source documents.

This module provides functions to format source documents retrieved from the
vector database into readable citations. Citations help users verify the
sources of information and increase trust in the RAG system's responses.

Key Functions:
    - format_citations_detailed(): Full citation with all metadata
    - format_citations_compact(): Compact citation for space-constrained displays

Example:
    >>> from src.rag.citations import format_citations_detailed
    >>>
    >>> # Get documents from retriever
    >>> docs = retriever.invoke("What is the Space Needle?")
    >>>
    >>> # Format as detailed citations
    >>> citations = format_citations_detailed(docs)
    >>> print(citations)
    ======================================================================
    📌 Source 1: Space Needle
    ======================================================================
    📍 Location: Seattle, Washington, United States of America
    📐 Coordinates: (47.620, -122.349)
    📄 Content:
    Name: Space Needle...

Note:
    These formatters expect Document objects with specific metadata fields:
    - name: Attraction name
    - city, state, country: Location information
    - lat, lon: Geographic coordinates
"""

from langchain_core.documents import Document


def format_citations_detailed(source_documents: list[Document]) -> str:
    """
    Format source documents into detailed, readable citations.

    Args:
        source_documents: List of Document objects from retriever or RAG chain

    Returns:
        Formatted citation string with full information including:
        - Source number and name
        - Location (city, state, country)
        - Geographic coordinates (if available)
        - Full document content

    Example:
        >>> docs = retriever.invoke("What is the Space Needle?")
        >>> citations = format_citations_detailed(docs)
        >>> print(citations)
        ======================================================================
        📌 Source 1: Space Needle
        ======================================================================

        📍 Location: Seattle, Washington, United States of America
        📐 Coordinates: (47.620, -122.349)
        📄 Content:
        Name: Space Needle...
    """
    if not source_documents:
        return "No sources available"

    citations = []

    for i, doc in enumerate(source_documents, 1):
        # Header
        name = doc.metadata.get("name", "Unknown")
        citation_parts = [f"\n{"=" * 70}", f"📌 Source {i}: {name}", f"{"=" * 70}"]

        # location info
        city = doc.metadata.get("city", "Unknown")
        state = doc.metadata.get("state", "Unknown")
        country = doc.metadata.get("country", "Unknown")
        citation_parts.append(f"\n📍 Location: {city}, {state}, {country}")

        # Coordinate info
        lat = doc.metadata.get("lat")
        lon = doc.metadata.get("lon")
        if lat and lon:
            citation_parts.append(f"📐 Coordinates: ({lat}, {lon})")

        citation_parts.append(f"📄 Content:\n{doc.page_content}")

        citations.append("\n".join(citation_parts))

    return "\n".join(citations)


def format_citations_compact(source_documents: list[Document]) -> str:
    """
    Format source documents in a compact style for space-constrained displays.

    Args:
        source_documents: List of Document objects

    Returns:
        Compact formatted citation string with:
        - Source number, name, and location
        - Brief content preview

    Example:
        >>> docs = retriever.invoke("What is the Space Needle?")
        >>> citations = format_citations_compact(docs)
        >>> print(citations)
        [1] Space Needle (Seattle, Washington)
            The Space Needle is an observation tower...

        [2] Pike Place Market (Seattle, Washington)
            Pike Place Market is a public market...
    """
    if not source_documents:
        return "No sources available"

    citations = []

    for i, doc in enumerate(source_documents, 1):
        name = doc.metadata.get("name", "Unknown")
        city = doc.metadata.get("city", "Unknown")
        state = doc.metadata.get("state", "Unknown")

        citation_parts = [
            f"[{i}] {name} ({city}, {state})",
            f"    {doc.page_content}",
        ]

        citations.append("\n".join(citation_parts))

    return "\n\n".join(citations)
