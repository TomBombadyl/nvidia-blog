# Configuration & Environment Analysis

## Overview

This document provides a comprehensive analysis of the configuration management system in the NVIDIA Blog MCP project, covering environment variables, default values, regional settings, and model configurations.

## Configuration File

**Location**: `mcp/config.py`

### Structure

The configuration is centralized in a single file that uses environment variables with sensible defaults.

## GCP Configuration

### Project ID

```python
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "nvidia-blog")
```

**Default**: `"nvidia-blog"`
**Environment Variable**: `GCP_PROJECT_ID`
**Usage**: GCP project identifier for all services
**Recommendation**: Always set via environment variable in production

### Region

```python
REGION = os.getenv("GCP_REGION", "europe-west3")
```

**Default**: `"europe-west3"` (Frankfurt)
**Environment Variable**: `GCP_REGION`
**Usage**: Primary region for RAG Corpus, Vector Search, GCS, Cloud Run
**Rationale**: 
- Central European location
- Low latency for European users
- Data residency compliance

### GCS Bucket

```python
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "nvidia-blogs-raw")
```

**Default**: `"nvidia-blogs-raw"`
**Environment Variable**: `GCS_BUCKET_NAME`
**Usage**: Storage bucket for raw XML, HTML, and cleaned text files
**Structure**: `{feed_folder}/{raw_xml|raw_html|clean}/{item_id}.{ext}`

## Vertex AI Configuration

### RAG Corpus

```python
RAG_CORPUS = os.getenv(
    "RAG_CORPUS",
    "projects/nvidia-blog/locations/europe-west3/"
    "ragCorpora/8496040697034440704"
)
```

**Default**: Full resource name with corpus ID
**Environment Variable**: `RAG_CORPUS`
**Format**: `projects/{project}/locations/{region}/ragCorpora/{corpus_id}`
**Usage**: RAG Corpus for semantic search
**Region**: `europe-west3` (matches REGION)

### Vector Search Endpoint

```python
VECTOR_SEARCH_ENDPOINT_ID = os.getenv(
    "VECTOR_SEARCH_ENDPOINT_ID", "8740721616633200640"
)
```

**Default**: `"8740721616633200640"`
**Environment Variable**: `VECTOR_SEARCH_ENDPOINT_ID`
**Usage**: Vector Search endpoint for query interface
**Region**: `europe-west3` (matches REGION)

### Vector Search Index

```python
VECTOR_SEARCH_INDEX_ID = os.getenv(
    "VECTOR_SEARCH_INDEX_ID", "1196910765810909184"
)
```

**Default**: `"1196910765810909184"`
**Environment Variable**: `VECTOR_SEARCH_INDEX_ID`
**Usage**: Vector Search index containing embeddings
**Note**: Comment indicates "blog_data_clean (Stream method for RAG Corpus)"

### Embedding Model

```python
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-multilingual-embedding-002")
```

**Default**: `"text-multilingual-embedding-002"`
**Environment Variable**: `EMBEDDING_MODEL`
**Specifications**:
- **Dimensions**: 768
- **Languages**: 50+ languages
- **Provider**: Google Vertex AI
**Usage**: Generates embeddings for Vector Search and RAG Corpus

## RSS Feed Configuration

### Feed Definitions

```python
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
```

**Structure**: Dictionary mapping feed names to configuration
**Fields**:
- **url**: RSS feed URL
- **folder**: GCS folder name for storage

**Feeds**:
1. **dev**: NVIDIA Developer Blog
   - URL: `https://developer.nvidia.com/blog/feed`
   - Folder: `dev`
2. **official**: NVIDIA Official Blog
   - URL: `https://feeds.feedburner.com/nvidiablog`
   - Folder: `official`

**Environment Variables**:
- `RSS_FEED_DEV`: Override Developer Blog URL
- `RSS_FEED_OFFICIAL`: Override Official Blog URL

## Processing Configuration

### Minimum Text Length

```python
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "10"))
```

**Default**: `10` characters
**Environment Variable**: `MIN_TEXT_LENGTH`
**Usage**: Minimum cleaned text length to process
**Rationale**: Filters out empty or near-empty content

### Maximum Retries

```python
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
```

**Default**: `3` attempts
**Environment Variable**: `MAX_RETRIES`
**Usage**: Maximum retry attempts for network operations
**Note**: Currently not used (tenacity handles retries)

### Request Timeout

```python
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
```

**Default**: `30` seconds
**Environment Variable**: `REQUEST_TIMEOUT`
**Usage**: HTTP request timeout
**Note**: Currently not used (timeouts set in code)

## RAG Query Configuration

### Vector Distance Threshold

```python
RAG_VECTOR_DISTANCE_THRESHOLD = float(os.getenv("RAG_VECTOR_DISTANCE_THRESHOLD", "0.7"))
```

**Default**: `0.7`
**Environment Variable**: `RAG_VECTOR_DISTANCE_THRESHOLD`
**Meaning**: Maximum cosine distance (1 - cosine similarity)
**Equivalent**: Minimum cosine similarity of 0.3
**Usage**: Filters low-similarity results from RAG Corpus
**Impact**:
- **Higher threshold**: More precise results (fewer false positives)
- **Lower threshold**: More results (fewer false negatives)
**Recommendation**: Tune based on precision/recall requirements

## Gemini Model Configuration

### Model Location

```python
GEMINI_MODEL_LOCATION = os.getenv("GEMINI_MODEL_LOCATION", "europe-west4")
```

**Default**: `"europe-west4"` (Netherlands)
**Environment Variable**: `GEMINI_MODEL_LOCATION`
**Usage**: Region for Gemini model API calls
**Rationale**:
- Closest European region to `europe-west3` (RAG Corpus)
- Meets Gemini availability requirements
- Minimizes latency while ensuring data residency

### Model Name

```python
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
```

**Default**: `"gemini-2.0-flash"`
**Environment Variable**: `GEMINI_MODEL_NAME`
**Usage**: Gemini model for query transformation and answer grading
**Characteristics**:
- **Speed**: Fast inference (optimized for latency)
- **Cost**: Cost-effective compared to larger models
- **Quality**: High quality for transformation and grading tasks

## Logging Configuration

### Log Level

```python
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

**Default**: `"INFO"`
**Environment Variable**: `LOG_LEVEL`
**Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
**Usage**: Controls logging verbosity
**Recommendation**: Use `DEBUG` for development, `INFO` for production

## Regional Architecture

### Multi-Region Strategy

The system uses a multi-region approach:

**europe-west3 (Frankfurt)**:
- RAG Corpus
- Vector Search
- GCS Bucket
- Cloud Run services (ingestion + MCP server)

**europe-west4 (Netherlands)**:
- Gemini 2.0 Flash models
- Query transformation
- Answer grading

**Rationale**:
- Minimizes latency between regions
- Ensures data residency compliance
- Meets service availability requirements
- Balances performance and compliance

### Region Selection Considerations

1. **Data Residency**: European regions for GDPR compliance
2. **Latency**: Closest regions to minimize API call latency
3. **Service Availability**: Regions where services are available
4. **Cost**: Regional pricing considerations

## Environment Variable Summary

| Variable | Default | Usage | Required |
|----------|---------|-------|----------|
| `GCP_PROJECT_ID` | `nvidia-blog` | GCP project identifier | Yes (production) |
| `GCP_REGION` | `europe-west3` | Primary region | Recommended |
| `GCS_BUCKET_NAME` | `nvidia-blogs-raw` | GCS bucket name | Recommended |
| `RAG_CORPUS` | Full resource name | RAG Corpus resource | Yes |
| `VECTOR_SEARCH_ENDPOINT_ID` | `8740721616633200640` | Vector Search endpoint | Optional |
| `VECTOR_SEARCH_INDEX_ID` | `1196910765810909184` | Vector Search index | Optional |
| `EMBEDDING_MODEL` | `text-multilingual-embedding-002` | Embedding model | Recommended |
| `RSS_FEED_DEV` | Developer Blog URL | Developer Blog feed | Optional |
| `RSS_FEED_OFFICIAL` | Official Blog URL | Official Blog feed | Optional |
| `MIN_TEXT_LENGTH` | `10` | Minimum text length | Optional |
| `MAX_RETRIES` | `3` | Max retry attempts | Optional |
| `REQUEST_TIMEOUT` | `30` | Request timeout | Optional |
| `RAG_VECTOR_DISTANCE_THRESHOLD` | `0.7` | Distance threshold | Recommended |
| `GEMINI_MODEL_LOCATION` | `europe-west4` | Gemini region | Recommended |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | Gemini model | Recommended |
| `LOG_LEVEL` | `INFO` | Logging level | Optional |

## Cloud Run Configuration

### Ingestion Job

**Environment Variables** (set in Cloud Run Job):
- `GCP_PROJECT_ID`: Project ID
- `GCP_REGION`: Region (europe-west3)
- `GCS_BUCKET_NAME`: Bucket name
- `RAG_CORPUS`: RAG Corpus resource name
- `VECTOR_SEARCH_ENDPOINT_ID`: Vector Search endpoint (optional)
- `VECTOR_SEARCH_INDEX_ID`: Vector Search index (optional)
- `EMBEDDING_MODEL`: Embedding model name
- `LOG_LEVEL`: Logging level

### MCP Server

**Environment Variables** (set in Cloud Build, `cloudbuild.mcp.yaml:42`):
```yaml
--set-env-vars=GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=europe-west3
```

**Set Variables**:
- `GCP_PROJECT_ID`: From Cloud Build `$PROJECT_ID`
- `GCP_REGION`: Hardcoded to `europe-west3`

**Other Variables**: Use defaults from `config.py`

## Configuration Best Practices

1. **Environment Variables**: Use environment variables for all production settings
2. **Defaults**: Sensible defaults for development/testing
3. **Documentation**: Document all configuration options
4. **Validation**: Validate configuration at startup
5. **Secrets**: Use Secret Manager for sensitive values (not currently used)

## Configuration Validation

### Current Validation

- **Type Conversion**: `int()`, `float()` for numeric values
- **Default Values**: Fallback to defaults if not set
- **No Explicit Validation**: No validation of ranges or formats

### Recommended Validation

1. **Region Validation**: Ensure region is valid GCP region
2. **Threshold Validation**: Ensure distance threshold is 0.0-1.0
3. **URL Validation**: Validate RSS feed URLs
4. **Resource Name Validation**: Validate RAG Corpus format
5. **Model Validation**: Ensure model names are valid

## Configuration Recommendations

1. **Secrets Management**: Use Secret Manager for sensitive values
2. **Configuration Validation**: Add startup validation
3. **Environment-Specific Configs**: Separate dev/staging/prod configs
4. **Documentation**: Document all configuration options
5. **Monitoring**: Monitor configuration changes
6. **Defaults Review**: Review and update defaults periodically

## Migration Considerations

### ID Migration

The system handles legacy ID formats automatically (`private/main.py:96-121`):
- Detects unsanitized IDs
- Migrates to sanitized format
- Updates processed_ids.json

### Configuration Migration

If configuration changes:
1. Update defaults in `config.py`
2. Update environment variables in Cloud Run
3. Test with new configuration
4. Monitor for issues

## Security Considerations

### Current Security

- **No Secrets**: No sensitive values in configuration
- **Service Accounts**: Uses Cloud Run service accounts
- **Public Access**: MCP server is publicly accessible (by design)

### Recommended Security

1. **Secrets Manager**: Use for any future sensitive values
2. **IAM Roles**: Ensure service accounts have minimal permissions
3. **Network Security**: Consider VPC for internal services
4. **Audit Logging**: Enable audit logs for configuration changes

