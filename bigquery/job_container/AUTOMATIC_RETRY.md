# Automatic Retry with Progressive Chunk Size Reduction

## Overview

The job now includes automatic retry mechanisms that ensure **no content is ever lost**. If processing fails due to token limits, the job automatically retries with progressively smaller chunk sizes until it succeeds.

## How It Works

### 1. Automatic Retry on Token Limit Errors

When a token limit error occurs, the job automatically retries with smaller chunk sizes:

1. **First attempt**: Uses adaptive chunk size based on content size
   - Normal (<200KB): 1000 tokens/chunk
   - Large (200-400KB): 750 tokens/chunk
   - Very large (>400KB): 500 tokens/chunk

2. **If token limit error**: Automatically retries with:
   - 500 tokens/chunk
   - 300 tokens/chunk
   - 200 tokens/chunk
   - 100 tokens/chunk (minimum)

3. **Never gives up**: Keeps trying until it works or reaches minimum size

### 2. Failed Item Detection and Retry

On each run, the job automatically detects items that:
- Exist in GCS (cleaned files)
- Are NOT in BigQuery (failed to process previously)

These items are automatically retried on the next run with the progressive chunk size reduction strategy.

### 3. Per-Item Isolation

Each blog post is processed independently:
- One failure doesn't stop other items
- Each item gets its own retry attempts
- Failed items are logged but don't block success

## Example Flow

### Scenario: 83KB Blog Post (Rubin Platform)

1. **First attempt**: 
   - Detects 83KB → uses 750-token chunks
   - Creates 18 chunks
   - Groups into batches
   - **If token limit error occurs...**

2. **Automatic retry**:
   - Retries with 500-token chunks
   - Creates more chunks (smaller size)
   - Creates more batches (more manageable)
   - **If still fails...**

3. **Continue retrying**:
   - 300-token chunks
   - 200-token chunks
   - 100-token chunks (guaranteed to work)

4. **Success**: Eventually processes with appropriate chunk size

### Scenario: Previously Failed Item

1. **Detection**: Job finds item in GCS but not in BigQuery
2. **Automatic retry**: Processes with progressive chunk size reduction
3. **Success**: Item gets indexed on retry

## Benefits

- ✅ **No content loss**: Every item eventually gets processed
- ✅ **Automatic**: No manual intervention needed
- ✅ **Resilient**: Handles edge cases automatically
- ✅ **Self-healing**: Failed items retry on next run
- ✅ **Efficient**: Uses smallest chunk size that works

## Logging

The job logs clearly show:
- Which attempt is being used
- Chunk size for each attempt
- Why retries are happening
- Success on which attempt

Example logs:
```
Attempt 1 for item_123: using 750-token chunks
Token limit error on attempt 1 for item_123. Retrying with smaller chunk size (500 tokens)...
Attempt 2 for item_123: using 500-token chunks
Successfully processed item_123 on attempt 2: 25/25 chunks (used 500-token chunks)
```

## Configuration

The progressive chunk sizes are:
- 1000 tokens (normal)
- 750 tokens (large)
- 500 tokens (very large)
- 300 tokens (retry 1)
- 200 tokens (retry 2)
- 100 tokens (retry 3 - minimum)

These can be adjusted in `process_item_with_retry()` if needed.

## Edge Cases Handled

1. **Token limit errors**: Automatic retry with smaller chunks
2. **Previously failed items**: Automatically detected and retried
3. **Very large content**: Progressive reduction until it works
4. **Partial failures**: Individual items fail but others continue
5. **Complete failures**: Only if all attempts exhausted

## Result

**No content is ever lost** - the job will keep trying until every item is processed or all retry strategies are exhausted.
