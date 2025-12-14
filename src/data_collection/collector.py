"""Data Collection Workflow"""

import json

from src.data_collection.geoapify_client import fetch_and_save_attractions
from src.utils.emoji_log import save


# =======================================
# 1. Get the attractions from Geoapify and filter them.
# =======================================
def collect_city_attraction(city_name, city_bbox, api_key, output_dir) -> list:
    """
    Collect attractions with Wikipedia links.

    Returns:
        list: Filtered attractions (only those with Wikipedia)
    """
    # 1. Get all attractions and saved them
    all_attractions = fetch_and_save_attractions(city_bbox=city_bbox, api_key=api_key)

    # 2. Filter out the attraction with wiki data
    filtered = filter_attractions_with_wikipedia(all_attractions)

    # 3. Save the filtered data
    output_file = output_dir / f"{city_name}_attractions_with_wikipedia.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    save(f"Filtered attractions saved at: {output_file.name}")

    return filtered


# =======================================
# 2. Filter attractions content wikipedia
# =======================================
def filter_attractions_with_wikipedia(features):
    """
    Filter attractions that have Wikipedia links.

    Args:
        features: List of GeoJSON features from Geoapify

    Returns:
        list: Features with Wikipedia data
    """
    return [
        feature
        for feature in features
        if feature.get("properties", {}).get("wiki_and_media", {}).get("wikipedia")
    ]
