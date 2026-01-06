# Resilient BigQuery Sync Job

This is a complete, production-ready implementation of the BigQuery sync job with:

- ✅ **Large content handling** - Uses dynamic batch sizing to handle oversized blog posts
- ✅ **Resilient error handling** - Continues processing other items if one fails
- ✅ **Partial success** - Exits successfully if any items are processed
- ✅ **Comprehensive logging** - Detailed logs for monitoring and debugging

## Features

1. **Automatic Batch Sizing**: Uses `LargeContentHandler` to split chunks into batches that respect the 20K token limit
2. **Adaptive Chunking**: Adjusts chunk size based on content size (normal/large/very large)
3. **Fallback Processing**: If batch processing fails, falls back to individual chunk processing
4. **Per-Item Isolation**: Each blog post is processed independently - one failure doesn't stop the job
5. **Smart Exit Codes**: Only fails if no items can be processed

## Building and Deploying

### Build the Docker Image

```bash
gcloud builds submit --config=bigquery/job_container/cloudbuild.yaml
```

### Manual Build (for testing)

```bash
docker build -f bigquery/job_container/Dockerfile -t bigquery-sync-job:test .
```

### Run Locally (for testing)

```bash
docker run --rm \
  -e GCP_PROJECT_ID=nvidia-blog \
  -e GCS_BUCKET_NAME=nvidia-blogs-raw \
  -e BIGQUERY_DATASET=nvidia_blog \
  -e EMBEDDING_MODEL=text-multilingual-embedding-002 \
  -e EMBEDDING_REGION=us-central1 \
  bigquery-sync-job:test
```

## How It Works

1. **Get Items to Process**: Finds new blog posts in GCS that aren't in BigQuery yet
2. **Load Content**: Downloads cleaned text files from GCS
3. **Determine Strategy**: Analyzes content size to choose optimal chunking strategy
4. **Chunk Content**: Splits content into chunks with overlap
5. **Create Batches**: Groups chunks into batches that respect token limits
6. **Generate Embeddings**: Processes batches with fallback to individual processing
7. **Store in BigQuery**: Saves items and chunks to BigQuery tables

## Error Handling

- **Individual Item Failure**: Logs error, continues with next item
- **Batch Failure**: Falls back to individual chunk processing
- **Chunk Failure**: Skips chunk, continues with other chunks
- **Complete Failure**: Only exits with error if no items processed at all

## Monitoring

The job logs:
- Total items found
- Items processed successfully
- Items that failed
- Detailed error messages
- Batch creation statistics

## Configuration

Environment variables:
- `GCP_PROJECT_ID`: GCP project ID (default: nvidia-blog)
- `GCS_BUCKET_NAME`: GCS bucket name (default: nvidia-blogs-raw)
- `BIGQUERY_DATASET`: BigQuery dataset (default: nvidia_blog)
- `EMBEDDING_MODEL`: Embedding model name (default: text-multilingual-embedding-002)
- `EMBEDDING_REGION`: Region for embedding API (default: us-central1)
