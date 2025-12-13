"""
Configuration file for Travel RAG project
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ===================================
# Project Paths
# ===================================
PROJECT_ROOT = Path(__file__).parent.parent  # Travel_rag/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ===================================
# Logger Settings
# ===================================
LOGGER_NAME = "travel_rag"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ===================================
# API Keys
# ===================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")

# ===================================
# LLM Settings
# ===================================
LLM_MODEL = "gemini-flash-latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2 "
TEMPERATURE = 0.7  # 0.0 = deterministic, 1.0 = creative

# ===================================
# RAG Settings
# ===================================
CHUNK_SIZE = 500  # tokens per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K = 3  # number of documents to retrieve

# ===================================
# Vector Database
# ===================================
USE_PINECONE = False  # False = ChromaDB (local), True = Pinecone (cloud)
COLLECTION_NAME = "global_attractions"

# ===================================
# Data Source Settings
# ===================================
GEOAPIFY_BASE_URL = "https://api.geoapify.com/v2/places"
TARGET_CITY = "Seattle"  # Starting city
CITY_BBOX = {  # Bounding box for New York
    "lon_min": -122.45,
    "lat_min": 47.48,
    "lon_max": -122.22,
    "lat_max": 47.73,
}
