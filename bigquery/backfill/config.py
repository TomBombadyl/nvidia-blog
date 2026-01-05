"""
Configuration for BigQuery backfill script.
"""

import os

# GCP Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "nvidia-blog")
REGION = os.getenv("GCP_REGION", "europe-west3")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "nvidia-blogs-raw")

# BigQuery Configuration
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "nvidia_blog")
BIGQUERY_TABLE_CHUNKS = os.getenv("BIGQUERY_TABLE_CHUNKS", "chunks")
BIGQUERY_TABLE_ITEMS = os.getenv("BIGQUERY_TABLE_ITEMS", "items")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-multilingual-embedding-002")

# RSS Feed Configuration (for reading from GCS)
RSS_FEEDS = {
    "dev": {"folder": "dev"},
    "official": {"folder": "official"}
}

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "768"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # Process items in batches

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

