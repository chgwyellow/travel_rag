"""
Main script to run the complete data collection pipeline.
This script orchestrates the entire data pipeline:
1. Collect attractions from Geoapify API
2. Enrich with Wikipedia descriptions
3. Build RAG-ready documents
"""

from pathlib import Path

from src.config import (
    CITY_BBOX,
    EMAIL,
    GEOAPIFY_API_KEY,
    PROCESSED_DATA_DIR,
    TARGET_CITY,
)
from src.data_collection.collector import collect_city_attraction
from src.data_collection.document_builder import batch_create_documents
from src.data_collection.enricher import enrich_attractions
from src.utils.emoji_log import done, error, info, success, task


def run_pipeline(
    city_name: str,
    city_bbox: dict,
    api_key: str,
    email: str,
    output_dir: Path,
):
    """
    Execute the complete data collection and enrichment pipeline.

    Args:
        city_name: Name of the city (default: from config)
        city_bbox: Bounding box coordinates (default: from config)
        api_key: Geoapify API key (default: from config)
        email: Email for Wikipedia User-Agent (default: from config)
        output_dir: Output directory (default: from config)

    Pipeline stages:
    1. Chapter 1: Collect attractions with Wikipedia links
    2. Chapter 2: Enrich with Wikipedia descriptions
    3. Build RAG documents
    """
    # Use defaults from config if not provided
    city_name = city_name or TARGET_CITY
    city_bbox = city_bbox or CITY_BBOX
    api_key = api_key or GEOAPIFY_API_KEY
    email = email or EMAIL
    output_dir = output_dir or PROCESSED_DATA_DIR

    try:
        task("Starting data collection pipeline...")

        # =======================================
        # 1: Data Collection
        # =======================================
        info("Data Collection")
        print("-" * 70)

        filtered_attractions = collect_city_attraction(
            city_name=city_name,
            city_bbox=city_bbox,
            api_key=api_key,
            output_dir=output_dir,
        )

        done(f"{len(filtered_attractions)} attractions collected")

        # =======================================
        # 2: Data Enrichment
        # =======================================
        info("Data Enrichment")
        print("-" * 70)

        enriched_attractions, stats = enrich_attractions(
            filtered_features=filtered_attractions,
            email=email,
            output_dir=output_dir,
        )

        done(f"{stats['complete_records']}/{stats['total']} complete")

        # =======================================
        # 3: Document Building
        # =======================================
        documents = batch_create_documents(
            attractions=enriched_attractions, output_dir=output_dir
        )

        done(f"{len(documents)} documents created")

        # =======================================
        # Pipeline Summary
        # =======================================
        info("Pipeline Summary")
        print("=" * 70)
        success(f"Total attractions processed: {len(filtered_attractions)}")
        success(f"Enriched with descriptions: {stats['with_description']}")
        success(f"RAG documents created: {len(documents)}")
        success(f"Data completeness: {stats['completeness_rate']}")
        print("=" * 70)

        done("Data pipeline completed successfully!")

    except Exception as e:
        error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_pipeline(
        city_name=TARGET_CITY,
        city_bbox=CITY_BBOX,
        api_key=GEOAPIFY_API_KEY,
        email=EMAIL,
        output_dir=PROCESSED_DATA_DIR,
    )
