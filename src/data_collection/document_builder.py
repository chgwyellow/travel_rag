"""RAG Document Builder"""

import json

from src.utils.emoji_log import done, info, save


# =======================================
# 1. Create single RAG document
# =======================================
def create_rag_document(attraction: dict) -> dict:
    """
    Format a single attraction into RAG document with complete metadata.

    Document format:
        Name: ...
        Location: ...
        Coordinates: ...
        Description: ...

    Args:
        attraction: Enriched attraction data with description and location

    Returns:
        dict: Document with place_id, name, document text, and complete metadata
    """
    # Extract basic fields
    name = attraction.get("name", "Unknown")
    address = attraction.get("address", "N/A")
    lat = attraction.get("lat", "N/A")
    lon = attraction.get("lon", "N/A")
    description = attraction.get("description", "No description available")

    # Format document text
    doc = f"""Name: {name}
Location: {address}
Coordinates: {lat}, {lon}
Description: {description}
"""

    # Extract metadata for vector database
    # Categories: keep as comma-separated string (ChromaDB doesn't support list)
    categories_str = attraction.get("category", "")

    # Extract country from address (last part after last comma)
    country = "Unknown"
    if address and address != "N/A":
        address_parts = address.split(",")
        if len(address_parts) > 0:
            country = address_parts[-1].strip()  # e.g., "United States of America"

    # Build complete metadata
    # Note: ChromaDB only supports str, int, float, bool, or None
    metadata = {
        "place_id": attraction.get("place_id"),
        "name": name,
        "city": attraction.get("city", "Unknown"),
        "state": attraction.get("state", "Unknown"),
        "country": country,
        "categories": categories_str,  # Keep as string, not list
        "lat": lat if lat != "N/A" else None,
        "lon": lon if lon != "N/A" else None,
        "has_description": bool(
            description and description != "No description available"
        ),
    }

    return {
        "place_id": attraction.get("place_id"),
        "name": name,
        "document": doc,
        "metadata": metadata,
    }


# =======================================
# 2. Batch create RAG documents
# =======================================
def batch_create_documents(attractions: list, output_dir) -> list:
    """
    Create RAG documents for all attractions and save to file.

    Args:
        attractions: List of enriched attractions
        output_dir: Directory to save documents

    Returns:
        list: List of formatted documents
    """
    info("Creating RAG documents...")

    documents = []

    for attraction in attractions:
        doc = create_rag_document(attraction=attraction)
        documents.append(doc)

    # Save documents
    city = attractions[0].get("city", "unknown")
    output_file = output_dir / f"{city}_attractions_documents.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    save(f"Documents saved to: {output_file.name}")

    done(f"Created {len(documents)} RAG documents")

    # Print sample
    info("\nSample document:")
    print("=" * 70)
    print(documents[0]["document"])
    print("=" * 70)

    return documents
