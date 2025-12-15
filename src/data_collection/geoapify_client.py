"""Geoapify API client"""

import json

import requests

from src.config import GEOAPIFY_BASE_URL, RAW_DATA_DIR
from src.utils.emoji_log import done, error, info, save, warn


def fetch_and_save_attractions(
    city_name: str,
    city_bbox: dict,
    api_key: str,
    categories: str = "tourism",
    limit: int = 500,
) -> list:
    """
    Fetch attractions from Geoapify API and save raw data to file.

    This function calls the Geoapify Places API to retrieve attractions within
    a specified bounding box and immediately saves the raw response to the
    data lake for data engineering best practices.

    Args:
        city_bbox: Dictionary containing bounding box coordinates with keys:
                   'lon_min', 'lat_min', 'lon_max', 'lat_max'
        api_key: Geoapify API key for authentication
        categories: Category filter for places (default: "tourism")
        limit: Maximum number of results to return (default: 500)

    Returns:
        list: List of GeoJSON features representing attractions

    Raises:
        requests.HTTPError: If API returns non-2xx status code
        requests.RequestException: If network request fails
        KeyError/IndexError: If response structure is unexpected

    Example:
        >>> bbox = {
        ...     "lon_min": -122.45,
        ...     "lat_min": 47.48,
        ...     "lon_max": -122.22,
        ...     "lat_max": 47.73
        ... }
        >>> attractions = fetch_and_save_attractions(bbox, "your_api_key")
        >>> print(f"Found {len(attractions)} attractions")

    Note:
        - Raw data is automatically saved to RAW_DATA_DIR
        - File is named as "{city}_attractions_raw.json"
        - City name is extracted from the first attraction's properties
    """

    # bbox format: rect:lon_min,lat_min,lon_max,lat_max
    filter_box = f"rect:{city_bbox["lon_min"]},{city_bbox["lat_min"]},{city_bbox["lon_max"]},{city_bbox["lat_max"]}"

    params = {
        "apiKey": api_key,
        "categories": categories,
        "filter": filter_box,
        "limit": 500,
    }

    # Check if raw data already exists
    raw_data_path = RAW_DATA_DIR / f"{city_name}_attractions_raw.json"

    if raw_data_path.exists():
        info(f"Loading existing raw data from {raw_data_path.name}")
        with open(raw_data_path, "r", encoding="utf-8") as f:
            attractions = json.load(f)
        done(f"Loaded {len(attractions)} attractions from cache")
        return attractions

    try:
        # Calling API
        info("Calling Geoapify API and save raw data...")
        response = requests.get(url=GEOAPIFY_BASE_URL, params=params, timeout=30)

        # Raise error
        response.raise_for_status()

        # Parse response
        raw_data = response.json()
        attractions = raw_data.get("features", [])

        # Check if we got any results
        if not attractions:
            warn("No attractions found in this area")
            warn(f"API response: {raw_data}")
            return []

        done(f"Found {len(attractions)} attractions")

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
