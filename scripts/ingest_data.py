"""
Main script to run the complete data collection pipeline.
This script orchestrates the entire data pipeline:
1. Collect attractions from Geoapify API
2. Enrich with Wikipedia descriptions
3. Build RAG-ready documents
"""

import argparse
import json
from pathlib import Path

from src.config import EMAIL, GEOAPIFY_API_KEY, PROCESSED_DATA_DIR
from src.data_collection.collector import collect_city_attraction
from src.data_collection.document_builder import batch_create_documents
from src.data_collection.enricher import enrich_attractions
from src.utils.emoji_log import done, error, info, success, task, warn


def load_cities_config(config_path: Path = None):
    """Load cities configuration from JSON file."""
    if config_path is None:
        config_file = Path("data/cities_config.json")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config["cities"]


def get_city_config(city_name: str):
    """Get configuration for a specific city."""
    cities = load_cities_config()

    for city in cities:
        if city["name"].lower() == city_name.lower():
            return city


def list_available_cities():
    """List all available cities from config."""
    cities = load_cities_config()

    info("Available cities:")
    print("=" * 70)

    for city in cities:
        status = "✅ Enabled" if city.get("enabled", True) else "❌ Disabled"
        print(f"  - {city['name']} ({city['country']}) - {status}")

    print("=" * 70)


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

    try:
        task(f"Starting data collection pipeline for {city_name}...")

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
    # Create CLI parser
    parser = argparse.ArgumentParser(
        description="Run data collection pipeline for a specific city"
    )
    parser.add_argument(
        "--city", type=str, help="City name (e.g., Seattle, Portland)", default=None
    )
    parser.add_argument("--list", action="store_true", help="List all available cities")
    args = parser.parse_args()

    # If user enters --list, show all cities and exit
    if args.list:
        list_available_cities()
        exit(0)

    # If user enters --city
    if args.city:
        # Find the specific city
        city_config = get_city_config(args.city)

        # If not found, show the error and exit
        if not city_config:
            error(f"City '{args.city}' not found in configuration")
            info("Use --list to see available cities")
            exit(1)

        # If the city is disabled, asking if continue
        if not city_config.get("enabled", True):
            warn(f"City '{args.city}' is disabled in configuration")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                exit(0)

        run_pipeline(
            city_name=city_config["name"],
            city_bbox=city_config["bbox"],
            api_key=GEOAPIFY_API_KEY,
            email=EMAIL,
            output_dir=PROCESSED_DATA_DIR,
        )

    # If there is no any parameters, show the help message and cities list
    else:
        parser.print_help()
        print("\n")
        list_available_cities()
