# Error Handling & Resilience Analysis

## Overview

The NVIDIA Blog MCP system implements comprehensive error handling and resilience patterns to ensure reliable operation in production. This document analyzes all error handling strategies, retry mechanisms, and graceful degradation patterns.

## Retry Strategies

### Tenacity Retry Framework

The system uses the `tenacity` library for consistent retry logic across all network operations.

#### RSS Fetching Retry

**Location**: `private/rss_fetcher.py:47-51`

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError))
)
def fetch_rss(self, url: str) -> bytes:
```

**Retry Configuration:**
- **Max Attempts**: 3
- **Backoff Strategy**: Exponential (2s, 4s, 8s)
- **Retry Conditions**: RequestException, ConnectionError
- **Total Wait Time**: Up to 14 seconds (2 + 4 + 8)

**Rationale:**
- Handles transient network errors
- Exponential backoff prevents overwhelming servers
- Limited attempts prevent infinite loops

#### HTML Fetching Retry

**Location**: `private/rss_fetcher.py:125-129`

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError))
)
def fetch_html(self, url: str) -> str:
```

**Same Configuration as RSS Fetching:**
- Consistent retry strategy across fetch operations
- Handles network timeouts and connection errors
- Prevents single point of failure

#### GCS Operations Retry

**Location**: `private/gcs_utils.py:29-33, 62-66, 85-89`

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,))
)
def read_json(self, blob_path: str) -> Optional[Dict]:
```

**Retry Configuration:**
- **Max Attempts**: 3
- **Backoff Strategy**: Exponential (2s, 4s, 8s)
- **Retry Conditions**: All exceptions (broad catch)
- **Rationale**: GCS operations can fail transiently

#### RAG Ingestion Retry

**Location**: `private/rag_ingest.py:66-70`

```python
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((Exception,))
)
def ingest_to_rag(self, text: str, metadata: Dict):
```

**Retry Configuration:**
- **Max Attempts**: 5 (more attempts for critical operation)
- **Backoff Strategy**: Exponential (5s, 10s, 20s, 40s, 60s)
- **Retry Conditions**: All exceptions
- **Total Wait Time**: Up to 135 seconds

**Rationale:**
- RAG ingestion is critical (more attempts)
- Longer backoff prevents overwhelming API
- Handles concurrent operation conflicts

#### Vector Search Retry

**Location**: `private/vector_search_ingest.py:74-78, 107-111`

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,))
)
def embed_text(self, text: str) -> List[float]:
```

**Retry Configuration:**
- **Max Attempts**: 3
- **Backoff Strategy**: Exponential (2s, 4s, 8s)
- **Retry Conditions**: All exceptions
- **Rationale**: Standard retry for API operations

## Concurrent Operation Handling

### RAG Corpus Concurrent Operations

**Location**: `private/rag_ingest.py:54-64, 127-135`

**Detection** (lines 54-64):

```python
def _should_retry_on_concurrent_operation(self, response: requests.Response) -> bool:
    """Check if error is due to concurrent operation and should be retried."""
    if response.status_code == 400:
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "")
            if "other operations running" in error_message.lower():
                return True
        except:
            pass
    return False
```

**Handling** (lines 127-135):

```python
# Handle concurrent operation errors with retry
if response.status_code == 400 and self._should_retry_on_concurrent_operation(response):
    error_data = response.json()
    error_msg = error_data.get("error", {}).get("message", "")
    logger.warning(
        f"Concurrent operation detected for {item_id}. "
        f"Will retry after backoff. Error: {error_msg}"
    )
    # Raise exception to trigger retry
    raise Exception(f"Concurrent operation: {error_msg}")
```

**Strategy:**
- Detects specific error message ("other operations running")
- Raises exception to trigger tenacity retry
- Exponential backoff prevents conflicts
- Logs warning for monitoring

**Benefits:**
- Prevents race conditions
- Handles parallel ingestion gracefully
- Automatic retry with backoff

## Graceful Degradation

### RAG Ingestion Failure Handling

**Location**: `private/main.py:203-214`

```python
# Ingest to RAG Corpus
log_structured("info", "Ingesting to RAG Corpus", feed=feed_name, item_id=item_id)
try:
    rag_ingester.ingest_to_rag(clean_text, metadata)
    log_structured("info", "RAG ingestion complete", feed=feed_name, item_id=item_id)
except Exception as e:
    # Log RAG ingestion errors but continue processing
    log_structured(
        "error",
        "RAG ingestion failed, continuing with other steps",
        feed=feed_name,
        item_id=item_id,
        error=str(e),
        error_type=type(e).__name__
    )
    # Don't raise - continue to mark as processed and update vector search
    # This allows the item to be tracked even if RAG fails
```

**Degradation Strategy:**
- Logs error but continues processing
- Marks item as processed (prevents reprocessing)
- Continues with Vector Search (if available)
- Prevents single failure from stopping pipeline

**Rationale:**
- RAG ingestion can fail (LRO timeouts, API errors)
- Item is still tracked and can be reprocessed
- Vector Search provides alternative search method

### Vector Search Optional Component

**Location**: `private/main.py:219-254`

**Initialization** (lines 309-334):

```python
# Initialize Vector Search ingester (optional - continue if it fails)
vector_ingester = None
try:
    log_structured("info", "Initializing Vector Search ingester", ...)
    vector_ingester = VectorSearchIngester(...)
    log_structured("info", "Vector Search ingester initialized successfully")
except Exception as e:
    log_structured(
        "warning",
        "Failed to initialize Vector Search ingester, continuing without it",
        error=str(e),
        error_type=type(e).__name__
    )
    logger.warning("Vector Search will be skipped. RAG ingestion will continue.")
```

**Usage** (lines 219-254):

```python
# Embed and upsert to Vector Search (if available)
if vector_ingester:
    try:
        log_structured("info", "Embedding text", ...)
        embedding = vector_ingester.embed_text(clean_text)
        log_structured("info", "Upserting vector", ...)
        vector_ingester.upsert_vector(embedding, item_id, metadata)
        log_structured("info", "Vector upsert complete", ...)
    except Exception as e:
        log_structured(
            "warning",
            "Vector Search upsert failed, continuing",
            feed=feed_name,
            item_id=item_id,
            error=str(e),
            error_type=type(e).__name__
        )
        # Continue processing even if vector search fails
else:
    log_structured("info", "Skipping Vector Search (not initialized)", ...)
```

**Degradation Strategy:**
- Vector Search is optional (not critical)
- Continues processing if initialization fails
- Continues processing if upsert fails
- Logs warnings for monitoring

**Rationale:**
- Vector Search provides alternative search method
- RAG Corpus is primary search method
- System functions without Vector Search

### Individual Item Failure Handling

**Location**: `private/main.py:261-264`

```python
except Exception as e:
    error_count += 1
    log_structured("error", "Error processing item", feed=feed_name, item_id=item_id, error=str(e), error_type=type(e).__name__)
    continue
```

**Strategy:**
- Catches all exceptions for individual items
- Logs error with context
- Continues processing next item
- Tracks error count for monitoring

**Benefits:**
- Prevents single item failure from stopping feed
- Allows partial success
- Provides error visibility

## Field Normalization

### API Response Variation Handling

**Location**: `mcp/mcp_server.py:47-74`

```python
@model_validator(mode='before')
@classmethod
def normalize_field_names(cls, data: Any) -> Dict[str, Any]:
    """
    Normalize field names from API response variations.
    Handles both 'text'/'content' and 'source_uri'/'uri' field name variations.
    """
    if not isinstance(data, dict):
        return data
    
    normalized = {}
    
    # Normalize text field: handle both 'text' and 'content'
    normalized['text'] = data.get('text') or data.get('content') or ""
    
    # Normalize source_uri field: handle both 'source_uri' and 'uri'
    normalized['source_uri'] = (
        data.get('source_uri') or 
        data.get('uri') or 
        None
    )
    
    # Preserve distance if present
    if 'distance' in data:
        normalized['distance'] = data['distance']
    
    return normalized
```

**Benefits:**
- Handles API response variations gracefully
- Prevents failures from API changes
- Maintains backward compatibility
- No code changes needed for API updates

### RAG API Response Handling

**Location**: `mcp/query_rag.py:176-245`

**Multiple Response Structure Handling:**

```python
# Try the expected structure first: contexts.contexts[]
contexts = result.get("contexts", {}).get("contexts", [])

# Fallback: if contexts is directly a list
if not contexts and isinstance(result.get("contexts"), list):
    contexts = result.get("contexts", [])

# Handle case where contexts might be wrapped in chunk objects
if contexts and isinstance(contexts[0], dict) and "chunk" in contexts[0]:
    contexts = [ctx.get("chunk", ctx) for ctx in contexts]

# Additional extraction: Check for nested text fields
# Try multiple possible text field locations
text = None
if ctx.get("text"):
    text = ctx.get("text")
elif ctx.get("content"):
    text = ctx.get("content")
elif isinstance(ctx.get("chunk"), dict) and ctx.get("chunk", {}).get("text"):
    text = ctx.get("chunk", {}).get("text")
# ... additional fallbacks
```

**Robustness:**
- Handles multiple API response structures
- Multiple fallback strategies
- Detailed debug logging
- Prevents failures from API changes

## Error Recovery Patterns

### LRO Timeout Handling

**Location**: `private/rag_ingest.py:154-219`

```python
max_poll_attempts = 120  # 10 minutes max (120 * 5 seconds)
poll_attempt = 0

while poll_attempt < max_poll_attempts:
    poll_attempt += 1
    
    # Get operation status
    op_result = op_response.json()
    
    if op_result.get("done", False):
        # Check for errors
        if "error" in op_result:
            error_details = op_result["error"]
            error_msg = (
                f"Import operation failed: {error_details.get('message', 'Unknown error')} "
                f"(code: {error_details.get('code', 'unknown')})"
            )
            logger.error(error_msg)
            raise Exception(error_msg)
        # ... success handling
        break
    
    time.sleep(5)  # Wait 5 seconds before polling again
else:
    # Max attempts reached
    error_msg = (
        f"Import operation timed out after {max_poll_attempts} attempts "
        f"({max_poll_attempts * 5} seconds)"
    )
    logger.error(error_msg)
    raise Exception(error_msg)
```

**Recovery Strategy:**
- 10-minute timeout prevents infinite loops
- Checks for operation errors
- Raises exception for retry mechanism
- Logs detailed error information

### ID Migration Handling

**Location**: `private/main.py:96-121`

```python
# Migrate: Convert any raw (unsanitized) IDs to sanitized format
# This handles legacy processed_ids.json files that may have raw URLs/guids
sanitized_ids = set()
migration_count = 0
for raw_id in raw_ids:
    # If the ID contains special chars that would be sanitized, it's a raw ID
    if re.search(r'[^\w\-_\.]', raw_id):
        sanitized = sanitize_item_id(raw_id)
        sanitized_ids.add(sanitized)
        migration_count += 1
    else:
        # Already sanitized
        sanitized_ids.add(raw_id)

existing_ids = sanitized_ids

# If we migrated any IDs, update the file
if migration_count > 0:
    processed_ids["ids"] = sorted(list(sanitized_ids))
    gcs.write_json(processed_ids_path, processed_ids)
    log_structured(
        "info",
        f"Migrated {migration_count} raw IDs to sanitized format",
        feed=feed_name,
        total_ids=len(sanitized_ids)
    )
```

**Recovery Strategy:**
- Detects legacy ID formats
- Migrates to sanitized format automatically
- Updates processed_ids.json file
- Prevents duplicate processing

## MCP Server Error Handling

### Tool Error Handling

**Location**: `mcp/mcp_server.py:336-342`

```python
except Exception as e:
    error_msg = f"Error searching NVIDIA blogs: {str(e)}"
    logger.exception("Error in search_nvidia_blogs tool")
    import traceback
    full_traceback = traceback.format_exc()
    logger.debug(f"Full traceback: {full_traceback}")
    return ErrorResult(error=error_msg)
```

**Strategy:**
- Catches all exceptions
- Returns ErrorResult (not raises exception)
- Logs full traceback for debugging
- Prevents server crashes

### Query Transformation Fallback

**Location**: `mcp/rag_query_transformer.py:144-150`

```python
# Fallback to original if transformation fails or is empty
if not transformed_query or len(transformed_query) < 3:
    logger.warning(
        "Query transformation returned empty result, "
        "using original query"
    )
    return original_query
```

**Strategy:**
- Falls back to original query on failure
- Logs warning for monitoring
- Prevents empty/invalid transformations
- Ensures query always proceeds

### Answer Grading Fallback

**Location**: `mcp/rag_answer_grader.py:198-213`

```python
except Exception as e:
    logger.error(f"Error grading contexts: {e}", exc_info=True)
    # Return a conservative grade that triggers refinement
    error_msg = f"Grading error: {type(e).__name__}: {str(e)}"
    if "Expecting value" in str(e) or "JSONDecodeError" in str(e):
        error_msg += " (Gemini returned invalid JSON - likely due to empty or malformed contexts)"
    
    return AnswerGrade(
        score=0.5,
        relevance=0.5,
        completeness=0.5,
        grounded=False,
        reasoning=error_msg,
        should_refine=True
    )
```

**Strategy:**
- Returns conservative grade on error
- Triggers refinement (should_refine=True)
- Provides error context in reasoning
- Prevents query failure

## Logging & Monitoring

### Structured Logging

**Location**: `private/main.py:63-71`

```python
def log_structured(level: str, message: str, **kwargs):
    """Emit structured JSON log entry."""
    log_entry = {
        "severity": level.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs
    }
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(log_entry))
```

**Benefits:**
- Structured format for parsing
- Includes context (feed, item_id, error details)
- Cloud Logging integration
- Easy filtering and alerting

### Error Logging

**Patterns:**
- **Structured Logging**: JSON format with context
- **Error Types**: Includes exception type names
- **Tracebacks**: Full tracebacks for debugging
- **Context**: Includes feed, item_id, error details

## Best Practices

1. **Retry Logic**: Consistent tenacity retry across all network operations
2. **Graceful Degradation**: Continues processing on non-critical failures
3. **Error Logging**: Comprehensive logging with context
4. **Field Normalization**: Handles API variations gracefully
5. **Timeout Handling**: Prevents infinite loops with timeouts
6. **Migration Support**: Handles legacy data formats automatically

## Recommendations

1. **Alerting**: Set up alerts for error rates and retry patterns
2. **Monitoring**: Track retry counts and failure rates
3. **Testing**: Add tests for error scenarios
4. **Documentation**: Document expected error scenarios
5. **Recovery**: Consider automatic retry for failed items

