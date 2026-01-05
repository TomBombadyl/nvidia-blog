# BigQuery MCP vs Original RAG MCP - Detailed Comparison

## ✅ Vector Database Configuration

### Embedding Model
- **Model**: `text-multilingual-embedding-002` ✓ (matches original)
- **Dimensions**: 768 ✓ (matches original)
- **Location**: `us-central1` ✓ (where model is available)

### BigQuery Vector Database Setup
- **Table**: `nvidia_blog.chunks`
- **Embedding Column**: `embedding ARRAY<FLOAT64>` (768 dimensions)
- **Vector Search**: `ML.DISTANCE()` function (cosine distance)
- **Total Chunks**: 403 with embeddings
- **Unique Items**: 147 articles
- **Feeds**: 2 (dev, official)

### Distance Calculation
- **Original RAG**: Vertex AI RAG Corpus API (cosine distance)
- **BigQuery**: `ML.DISTANCE(embedding, query_embedding)` (cosine distance) ✓
- **Threshold**: `0.7` ✓ (matches original)

## ✅ Retrieval Method Comparison

### Query Flow
Both implementations follow the **exact same flow**:

1. **Query Transformation** (if enabled)
   - Uses same `QueryTransformer` class
   - Same Gemini model (`gemini-2.0-flash`)
   - Same location (`europe-west4`)

2. **Vector Search**
   - **Original**: RAG Corpus API `retrieveContexts`
   - **BigQuery**: SQL `ML.DISTANCE()` query
   - Both use cosine distance
   - Both filter by `vector_distance_threshold`

3. **Answer Grading** (if enabled)
   - Uses same `AnswerGrader` class
   - Same Gemini model and location
   - Same `min_acceptable_score = 0.6`

4. **Iterative Refinement**
   - Same `max_refinement_iterations = 2`
   - Same refinement prompt logic
   - Same best-result tracking

### Header Filtering (NOW MATCHES)
- **Original**: Filters out header-only chunks (< 100 chars after removing metadata)
- **BigQuery**: ✅ **NOW IMPLEMENTED** - Same filtering logic
- Removes: "Publication Date:", "Title:", "Source:", "---" separators
- Keeps only chunks with ≥100 characters of actual content

### Context Format
Both return contexts with:
- `text`: Chunk text content
- `source_uri`: Source URL
- `distance`: Similarity distance (lower = more similar)
- Additional metadata: `chunk_id`, `item_id`, `feed`, `chunk_index`, `publication_date`, `title`

## ✅ Configuration Matching

| Setting | Original MCP | BigQuery MCP | Match |
|---------|-------------|--------------|-------|
| Embedding Model | `text-multilingual-embedding-002` | `text-multilingual-embedding-002` | ✅ |
| Embedding Dimensions | 768 | 768 | ✅ |
| Distance Threshold | 0.7 | 0.7 | ✅ |
| Query Transformer | Enabled | Enabled | ✅ |
| Answer Grader | Enabled | Enabled | ✅ |
| Max Refinements | 2 | 2 | ✅ |
| Gemini Model | `gemini-2.0-flash` | `gemini-2.0-flash` | ✅ |
| Gemini Location | `europe-west4` | `europe-west4` | ✅ |
| Header Filtering | Yes | ✅ **NOW YES** | ✅ |

## ✅ Query Ordering

### Original RAG Corpus
- Orders by similarity (distance ASC)
- No explicit date sorting

### BigQuery
- Orders by similarity (distance ASC) ✓
- Secondary sort by `publication_date DESC NULLS LAST` (when dates available)
- **Note**: Currently all `publication_date` are NULL (data issue, not retrieval issue)

## ⚠️ Known Data Issues

1. **Publication Dates**: All NULL in BigQuery
   - **Cause**: Backfill doesn't extract dates from cleaned text
   - **Impact**: Date sorting doesn't work (but similarity sorting does)
   - **Fix Needed**: Update backfill/incremental jobs to extract dates

2. **Data Volume**: 403 chunks vs likely more in RAG Corpus
   - **Cause**: Backfill may not have processed all items
   - **Impact**: Fewer results available
   - **Fix Needed**: Verify backfill completed or re-run

## ✅ Vector Database Verification

### BigQuery as Vector DB
- ✅ Uses `ML.DISTANCE()` for cosine similarity search
- ✅ Embeddings stored as `ARRAY<FLOAT64>` (768 dimensions)
- ✅ Table partitioned by date for performance
- ✅ Clustered by `feed, item_id, chunk_index` for query optimization
- ✅ All 403 chunks have valid 768-dimensional embeddings

### Performance Characteristics
- **Query Latency**: BigQuery ML.DISTANCE is optimized for vector search
- **Scalability**: BigQuery handles millions of vectors efficiently
- **Cost**: Pay-per-query (no index maintenance costs)

## Summary

✅ **Retrieval method now matches original MCP exactly:**
- Same query transformation
- Same vector search (cosine distance)
- Same answer grading
- Same iterative refinement
- **Same header filtering** (just added)

✅ **BigQuery is properly configured as vector database:**
- 768-dimensional embeddings stored correctly
- ML.DISTANCE() function working
- All chunks have valid embeddings

⚠️ **Data issues to fix:**
- Publication dates need to be extracted during backfill
- Verify all items were processed

## Conclusion

**Status Update (v2.0.0)**: The BigQuery MCP implementation has been renamed to `nvidia-blog-mcp` and is now the **primary production service**. The original RAG Corpus-based implementation has been archived to `archive/mcp-rag-corpus/` for reference.

**Why BigQuery MCP is Primary:**
- ✅ Date-aware filtering capabilities (extract and filter by publication dates)
- ✅ SQL-level date queries for efficient temporal searches
- ✅ Same query transformation and grading quality as original
- ✅ Improved performance with BigQuery ML.DISTANCE
- ✅ Better integration with existing BigQuery data pipeline

**Migration Notes:**
- Service URL changed from `nvidia-blog-mcp-server` to `nvidia-blog-mcp`
- All client configurations should be updated to use the new service name
- The archived RAG Corpus implementation remains available for reference but is no longer actively maintained
