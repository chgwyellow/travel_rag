"""Wikipedia API client"""

import time

import requests

from src.config import WIKIPEDIA_API_BASE
from src.utils.emoji_log import error, success


# =======================================
# 1. Extract attractions data
# =======================================
def extract_attraction_data(feature: dict) -> dict:
    """
    Extract and structure attraction data from Geoapify feature.

    Args:
        feature: Raw GeoJSON feature from Geoapify API

    Returns:
        dict: Structured attraction data with standardized fields
    """
    props = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates", [None, None])

    return {
        "place_id": props.get("place_id"),
        "name": props.get("name"),
        "category": ", ".join(props.get("categories", [])),
        "address": props.get("formatted"),
        "address_line1": props.get("address_line1"),
        "address_line2": props.get("address_line2"),
        "lon": coords[0],
        "lat": coords[1],
        "city": props.get("city"),
        "state": props.get("state"),
        "postcode": props.get("postcode"),
        "datasource": props.get("datasource", {}).get("sourcename"),
        "wiki_code": props.get("wiki_and_media", {}).get("wikipedia"),
    }


# =======================================
# 2. Fetch the wiki description
# =======================================
def fetch_description(wiki_code: str, email: str) -> str | None:
    """
    Fetch Wikipedia description for a single attraction.

    Args:
        wiki_code: Wikipedia code in format "language:title" (e.g., "en:Seattle Central Library")
        email: Email address for User-Agent header

    Returns:
        str | None: Description text if found, None otherwise

    Raises:
        Does not raise exceptions - returns None on any error
    """

    try:
        language, title = wiki_code.split(":", maxsplit=1)
        language = language.strip()
        title = title.strip()

        # API URL
        api_url = WIKIPEDIA_API_BASE.format(language=language)

        # Headers
        headers = {"User-Agent": f"TravelRAG/1.0 (Educational Project; {email})"}

        # Params
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": 1,
        }

        response = requests.get(url=api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse response
        wiki_data = response.json()
        pages = wiki_data.get("query", {}).get("pages", {})

        # Extract description
        for page_id, page_data in pages.items():
            if "extract" in page_data and "missing" not in page_data:
                return page_data["extract"]

        return None

    except requests.exceptions.HTTPError as e:
        error(f"HTTP Error: {e.response.status_code}")
        return None

    except requests.exceptions.RequestException as e:
        error(f"Request failed: {str(e)}")
        return None

    except Exception as e:
        error(f"Unexpected error: {str(e)}")
        return None


# =======================================
# 3. Batch fetch descriptions
# =======================================
def batch_fetch_descriptions(
    features: list, email: str, rate_limit: float = 0.5
) -> list:
    """
    Fetch Wikipedia descriptions for multiple attractions with rate limiting.

    Args:
        features: List of Geoapify raw features
        email: Email address for User-Agent header
        rate_limit: Delay in seconds between API requests (default: 0.5)

    Returns:
        list: List of attractions with descriptions added

    Note:
        - Attractions without wiki_code will have description set to None
        - Failed API calls will result in description set to None
        - Progress is logged for each attraction
    """
    attraction_list = []
    success_count = 0
    error_count = 0

    for feature in features:
        # Extract data
        attraction = extract_attraction_data(feature)

        # Check the wiki_code
        wiki_code = attraction.get("wiki_code")
        if wiki_code:
            # Obtain description
            description = fetch_description(wiki_code=wiki_code, email=email)
            attraction["description"] = description

            if description and attraction["name"]:
                success_count += 1
                success(f"{attraction['name']}, success counts: {success_count}")
                attraction_list.append(attraction)
            else:
                error_count += 1
                error(
                    f"{attraction['name']} - No Name or description, error counts: {error_count}"
                )

            # Rate limiting
            time.sleep(rate_limit)

    return attraction_list
