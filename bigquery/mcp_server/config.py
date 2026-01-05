"""
Configuration for BigQuery MCP Server.
"""

import os
import sys

# Add parent directory to path to import from existing mcp/ if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# GCP Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "nvidia-blog")
REGION = os.getenv("GCP_REGION", "europe-west3")

# BigQuery Configuration
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "nvidia_blog")
BIGQUERY_TABLE_CHUNKS = os.getenv("BIGQUERY_TABLE_CHUNKS", "chunks")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-multilingual-embedding-002")

# RAG Query Configuration
RAG_VECTOR_DISTANCE_THRESHOLD = float(os.getenv("RAG_VECTOR_DISTANCE_THRESHOLD", "0.7"))

# Gemini Model Configuration (for query transformation and grading)
GEMINI_MODEL_LOCATION = os.getenv("GEMINI_MODEL_LOCATION", "europe-west4")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

