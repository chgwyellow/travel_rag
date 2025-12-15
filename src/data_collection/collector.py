"""Data Collection Workflow"""

import json

from src.data_collection.geoapify_client import fetch_and_save_attractions
from src.utils.emoji_log import save, warn


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
    all_attractions = fetch_and_save_attractions(
        city_name=city_name, city_bbox=city_bbox, api_key=api_key
    )

    # 2. Filter out the attraction with wiki data
    filtered = filter_attractions_with_wikipedia(all_attractions)

    # 3. Remove duplicates based on place_id
    unique_attractions = remove_duplicates(filtered)

    # 4. Save the filtered data
    output_file = output_dir / f"{city_name}_attractions_with_wikipedia.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_attractions, f, indent=2, ensure_ascii=False)

    save(f"Filtered attractions saved at: {output_file.name}")

    return unique_attractions


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


# =======================================
# 3. Remove duplicates based on place_id
# =======================================
def remove_duplicates(features: list) -> list:
    """
    Remove duplicate attractions based on wiki_code.

    Since all features have been filtered to have Wikipedia links,
    we use wiki_code as the primary deduplication key.

    Args:
        features: List of GeoJSON features (all with Wikipedia links)

    Returns:
        list: Features without duplicates
    """
    seen_wiki_codes = set()
    unique_features = []
    duplicates_info = []

    for feature in features:
        props = feature.get("properties", {})
        wiki_code = props.get("wiki_and_media", {}).get("wikipedia", "")
        name = props.get("name", "")

        # Skip the wiki-codeless attraction
        if not wiki_code:
            continue

        # check wiki_code
        if wiki_code in seen_wiki_codes:
            duplicates_info.append(f"{name} (duplicate wiki: {wiki_code})")
            continue

        # Add to unique list
        seen_wiki_codes.add(wiki_code)
        unique_features.append(feature)

    # Log duplicate details
    if duplicates_info:
        warn(f"Found {len(duplicates_info)} duplicates:")
        for dup in duplicates_info:
            warn(f" - {dup}")

    return unique_features
