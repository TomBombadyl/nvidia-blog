"""
Configuration management for RSS ingestion pipeline.
Uses environment variables with sensible defaults.
"""

import os
from typing import Dict

# GCP Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "nvidia-blog")
REGION = os.getenv("GCP_REGION", "europe-west3")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "nvidia-blogs-raw")

# Vertex AI Configuration
RAG_CORPUS = os.getenv(
    "RAG_CORPUS",
    "projects/nvidia-blog/locations/europe-west3/"
    "ragCorpora/8070450532247928832"
)
VECTOR_SEARCH_ENDPOINT_ID = os.getenv(
    "VECTOR_SEARCH_ENDPOINT_ID", "8740721616633200640"
)
VECTOR_SEARCH_INDEX_ID = os.getenv(
    "VECTOR_SEARCH_INDEX_ID", "3602747760501063680"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# RSS Feed Configuration
RSS_FEEDS: Dict[str, Dict[str, str]] = {
    "dev": {
        "url": os.getenv(
            "RSS_FEED_DEV", "https://developer.nvidia.com/blog/feed"
        ),
        "folder": "dev"
    },
    "official": {
        "url": os.getenv(
            "RSS_FEED_OFFICIAL", "https://feeds.feedburner.com/nvidiablog"
        ),
        "folder": "official"
    }
}

# Processing Configuration
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# RAG Query Configuration
RAG_VECTOR_DISTANCE_THRESHOLD = float(os.getenv("RAG_VECTOR_DISTANCE_THRESHOLD", "0.5"))

# Gemini Model Configuration
# Gemini models are available in europe-west4 (Netherlands) - closest to europe-west3 (Frankfurt)

GEMINI_MODEL_LOCATION = os.getenv("GEMINI_MODEL_LOCATION", "europe-west4")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

