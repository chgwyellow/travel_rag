"""Geoapify API client"""

import json

import requests

from src.config import GEOAPIFY_BASE_URL, RAW_DATA_DIR
from src.utils.emoji_log import done, error, save


def fetch_and_save_attractions(
    city_bbox: dict,
    api_key: str,
    categories: str = "tourism",
    limit: int = 500,
):
    """Calling Geoapify API and save raw data"""

    # bbox format: rect:lon_min,lat_min,lon_max,lat_max
    filter_box = f"rect:{city_bbox["lon_min"]},{city_bbox["lat_min"]},{city_bbox["lon_max"]},{city_bbox["lat_max"]}"

    params = {
        "apiKey": api_key,
        "categories": categories,
        "filter": filter_box,
        "limit": 500,
    }

    try:
        # Calling API
        response = requests.get(url=GEOAPIFY_BASE_URL, params=params, timeout=30)

        # Raise error
        response.raise_for_status()

        # Parse response
        raw_data = response.json()
        attractions = raw_data["features"]
        city = attractions[0]["properties"]["city"]
        done(f"Found {len(attractions)} attractions")

        raw_data_path = RAW_DATA_DIR / f"{city}_attractions_raw.json"

        with open(raw_data_path, "w", encoding="utf-8") as f:
            json.dump(attractions, f, indent=2, ensure_ascii=False)

        save(f"Raw data saved: {raw_data_path}")

        return attractions
    except requests.exceptions.HTTPError as e:
        error(f"HTTP Error: {e.response.status_code}")
        error(f"Response: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        error(f"Request failed: {str(e)}")
        raise
    except (KeyError, IndexError) as e:
        error(f"Unexpected response structure: {str(e)}")
        raise
