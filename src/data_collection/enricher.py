"""Data Enrichment Workflow"""

import json

from src.data_collection.wikipedia_client import batch_fetch_descriptions
from src.utils.emoji_log import done, info, save


# =======================================
# 1. Add the description to attractions and produce the stats
# =======================================
def enrich_attractions(
    filtered_features, email, output_dir
) -> tuple:
    """
    Enrich attractions with Wikipedia descriptions and location data.

    1. Fetch Wikipedia descriptions for all attractions
    2. Merge location data from raw Geoapify response
    3. Validate data quality
    4. Save enriched data

    Args:
        filtered_features: Filtered attractions from collector (with Wikipedia)
        raw_features: Raw Geoapify features (for location data)
        email: Email address for Wikipedia API User-Agent
        output_dir: Directory to save enriched data

    Returns:
        tuple: (enriched_attractions, quality_stats)
    """
    info("Starting data enrichment workflow...")

    # 1. Fetch Wikipedia descriptions
    info("Fetching Wikipedia descriptions...")
    enriched = batch_fetch_descriptions(features=filtered_features, email)

    # 2. Validate data quality
    info("Validating data quality...")
    stats = validate_data_quality(enriched)

    # 3. Save enriched data
    city = enriched[0].get("city", "")
    output_file = output_dir / f"{city}_attractions_enriched.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    return enriched, stats


# =======================================
# 2. Validate data quality and return the stats
# =======================================
def validate_data_quality(attractions) -> dict:
    """
    Validate data quality and return statistics.

    Args:
        attractions: List of enriched attractions

    Returns:
        dict: Quality statistics
    """
    total = len(attractions)
    with_description = sum(1 for a in attractions if a.get("description"))
    complete = sum(
        1
        for a in attractions
        if a.get("name") and a.get("description") and a.get("lon")
    )

    stats = {
        "total": total,
        "with_description": with_description,
        "complete_records": complete,
        "completeness_rate": f"{(complete/total*100):.1f}%" if total > 0 else "0%",
    }

    # Print statistics
    info("Data Quality Statistics:")
    print(f"  - Total attractions: {stats['total']}")
    print(f"  - With descriptions: {stats['with_description']}")
    print(
        f"  - Complete records: {stats['complete_records']} ({stats['completeness_rate']})"
    )

    return stats
