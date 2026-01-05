-- BigQuery Schema Setup for NVIDIA Blog MCP
-- Creates dataset and tables for storing blog content chunks with embeddings

-- Create dataset
CREATE SCHEMA IF NOT EXISTS `nvidia_blog`
OPTIONS(
  description="NVIDIA Blog content and embeddings for BigQuery MCP server",
  location="europe-west3"
);

-- Table for blog items (articles)
CREATE TABLE IF NOT EXISTS `nvidia_blog.items` (
  item_id STRING NOT NULL,
  feed STRING NOT NULL,
  source_uri STRING NOT NULL,
  title STRING,
  publication_date TIMESTAMP,
  processed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY feed, item_id
OPTIONS(
  description="Blog article metadata"
);

-- Table for text chunks with embeddings
CREATE TABLE IF NOT EXISTS `nvidia_blog.chunks` (
  chunk_id STRING NOT NULL,
  item_id STRING NOT NULL,
  feed STRING NOT NULL,
  source_uri STRING NOT NULL,
  text STRING NOT NULL,
  embedding ARRAY<FLOAT64>,  -- 768 dimensions (NOT NULL not allowed on ARRAY in BigQuery)
  chunk_index INT64 NOT NULL,  -- Order within article
  chunk_size_words INT64,  -- Number of words in chunk
  publication_date TIMESTAMP,
  title STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY feed, item_id, chunk_index
OPTIONS(
  description="Text chunks with embeddings for semantic search"
);

-- Create primary key constraints (BigQuery doesn't enforce but documents intent)
-- item_id is unique per feed in items table
-- chunk_id is unique in chunks table

