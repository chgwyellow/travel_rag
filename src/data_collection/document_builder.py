"""RAG Document Builder"""

import json

from src.utils.emoji_log import done, info, save


# =======================================
# 1. Create single RAG document
# =======================================
def create_rag_document(attraction: dict) -> dict:
    """
    Format a single attraction into RAG document.

    Document format:
        Name: ...
        Location: ...
        Coordinates: ...
        Description: ...

    Args:
        attraction: Enriched attraction data with description and location

    Returns:
        dict: Document with place_id, name, and formatted document text
    """
    name = attraction.get("name", "Unknown")
    address = attraction.get("address", "N/A")
    lat = attraction.get("lat", "N/A")
    lon = attraction.get("lon", "N/A")
    description = attraction.get("description", "No description available")

    # Format document
    doc = f"""Name: {name}
Location: {address}
Coordinates: {lat}, {lon}
Description: {description}
"""

    return {
        "place_id": attraction.get("plac_id"),
        "name": attraction.get("name"),
        "document": doc,
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
    documents[0]["document"]
    print("=" * 70)

    return documents
