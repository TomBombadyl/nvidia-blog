"""
Resilient BigQuery Sync Job

Processes blog posts from GCS storage and syncs them to BigQuery.
Implements:
- Large content handling with dynamic batch sizing
- Resilient error handling (continues on individual failures)
- Partial success reporting
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from google.cloud import storage, bigquery
from vertexai.language_models import TextEmbeddingModel

# Import large content handler (copied to same directory in Docker)
from large_content_handler import LargeContentHandler, ChunkingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'nvidia-blog')
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'nvidia-blogs-raw')
DATASET_ID = os.getenv('BIGQUERY_DATASET', 'nvidia_blog')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-multilingual-embedding-002')
EMBEDDING_REGION = os.getenv('EMBEDDING_REGION', 'us-central1')

# BigQuery table names
ITEMS_TABLE = 'items'
CHUNKS_TABLE = 'chunks'


class ResilientBigQuerySync:
    """BigQuery sync job with resilient error handling."""
    
    def __init__(self):
        """Initialize sync job."""
        self.storage_client = storage.Client(project=PROJECT_ID)
        self.bq_client = bigquery.Client(project=PROJECT_ID)
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        self.content_handler = LargeContentHandler()
        
        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        logger.info(f"Initialized BigQuery sync job")
        logger.info(f"Project: {PROJECT_ID}, Bucket: {BUCKET_NAME}, Dataset: {DATASET_ID}")
    
    def get_processed_item_ids(self, feed: str) -> set:
        """Get set of item IDs already in BigQuery."""
        try:
            query = f"""
            SELECT DISTINCT item_id
            FROM `{PROJECT_ID}.{DATASET_ID}.{ITEMS_TABLE}`
            WHERE feed = @feed
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("feed", "STRING", feed)
                ]
            )
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            item_ids = {row.item_id for row in results}
            logger.info(f"Found {len(item_ids)} processed items for feed: {feed}")
            return item_ids
        except Exception as e:
            logger.warning(f"Error getting processed items: {e}, assuming none processed")
            return set()
    
    def get_failed_items_from_previous_run(self, feed: str) -> List[str]:
        """
        Get items that failed in previous runs but are still in GCS.
        These should be retried with smaller chunk sizes.
        """
        # Check for items in GCS that aren't in BigQuery
        # This will catch items that failed previously
        bucket = self.storage_client.bucket(BUCKET_NAME)
        prefix = f"{feed}/clean/"
        
        blobs = bucket.list_blobs(prefix=prefix)
        gcs_item_ids = set()
        for blob in blobs:
            if blob.name.endswith('.txt'):
                filename = blob.name.split('/')[-1]
                item_id = filename.replace('.txt', '')
                gcs_item_ids.add(item_id)
        
        # Get items already in BigQuery
        processed_ids = self.get_processed_item_ids(feed)
        
        # Items in GCS but not in BigQuery = failed items to retry
        failed_items = list(gcs_item_ids - processed_ids)
        
        if failed_items:
            logger.info(
                f"Found {len(failed_items)} items in GCS but not in BigQuery for feed {feed}. "
                f"These will be retried with automatic chunk size reduction."
            )
        
        return failed_items
    
    def get_items_to_process(self, feed: str) -> List[str]:
        """
        Get list of item IDs that need processing.
        Includes both new items and previously failed items that should be retried.
        """
        # Get items that failed in previous runs (in GCS but not in BigQuery)
        failed_items = self.get_failed_items_from_previous_run(feed)
        
        # Also check for any new items (though get_failed_items already covers this)
        # This ensures we don't miss anything
        bucket = self.storage_client.bucket(BUCKET_NAME)
        prefix = f"{feed}/clean/"
        
        blobs = bucket.list_blobs(prefix=prefix)
        all_item_ids = set()
        for blob in blobs:
            if blob.name.endswith('.txt'):
                filename = blob.name.split('/')[-1]
                item_id = filename.replace('.txt', '')
                all_item_ids.add(item_id)
        
        # Filter out already processed items
        processed_ids = self.get_processed_item_ids(feed)
        items_to_process = list(all_item_ids - processed_ids)
        
        logger.info(
            f"Feed {feed}: {len(items_to_process)} items to process "
            f"({len(failed_items)} from previous failures, "
            f"{len(items_to_process) - len(failed_items)} new) "
            f"out of {len(all_item_ids)} total"
        )
        return items_to_process
    
    def load_item_content(self, item_id: str, feed: str) -> Optional[str]:
        """Load item content from GCS."""
        try:
            bucket = self.storage_client.bucket(BUCKET_NAME)
            blob_path = f"{feed}/clean/{item_id}.txt"
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning(f"Blob not found: {blob_path}")
                return None
            
            content = blob.download_as_text()
            return content
        except Exception as e:
            logger.error(f"Error loading content for {item_id}: {e}")
            return None
    
    def parse_item_metadata(self, content: str) -> Dict:
        """Parse metadata from item content."""
        metadata = {
            'publication_date': None,
            'title': None,
            'source': None
        }
        
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('Publication Date:'):
                date_str = line.replace('Publication Date:', '').strip()
                try:
                    metadata['publication_date'] = datetime.fromisoformat(
                        date_str.replace('Z', '+00:00')
                    )
                except:
                    pass
            elif line.startswith('Title:'):
                metadata['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Source:'):
                metadata['source'] = line.replace('Source:', '').strip()
        
        return metadata
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Chunk content into smaller pieces."""
        # Remove metadata header
        lines = content.split('\n')
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '---':
                content_start = i + 1
                break
        
        text_content = '\n'.join(lines[content_start:])
        
        # Simple chunking (can be improved with better text splitting)
        chunks = []
        words = text_content.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size * 4:  # ~4 chars per token
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'chunk_index': len(chunks)
                    })
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_index': len(chunks)
            })
        
        return chunks
    
    def generate_embeddings_batch(self, chunks: List[Dict]) -> List[List[float]]:
        """Generate embeddings for a batch of chunks."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.get_embeddings(texts)
        return embeddings
    
    def generate_embedding(self, chunk: Dict) -> List[float]:
        """Generate embedding for a single chunk."""
        embedding = self.embedding_model.get_embeddings([chunk['text']])[0]
        return embedding
    
    def is_token_limit_error(self, error: Exception) -> bool:
        """Check if error is due to token limit exceeded."""
        error_str = str(error).lower()
        return (
            'token' in error_str and 
            ('limit' in error_str or 'exceeded' in error_str or '20000' in error_str)
        )
    
    def process_batches_with_fallback(
        self, item_id: str, batches: List[List[Dict]]
    ) -> List[Dict]:
        """Process batches with fallback to individual processing."""
        processed_chunks = []
        
        for batch_idx, batch in enumerate(batches):
            try:
                logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)} for {item_id}")
                embeddings = self.generate_embeddings_batch(batch)
                
                for chunk, embedding in zip(batch, embeddings):
                    chunk['embedding'] = embedding
                    processed_chunks.append(chunk)
                    
            except Exception as batch_error:
                is_token_error = self.is_token_limit_error(batch_error)
                if is_token_error:
                    logger.warning(
                        f"Batch {batch_idx + 1} failed due to token limit for {item_id}. "
                        f"Falling back to individual processing."
                    )
                else:
                    logger.warning(
                        f"Batch {batch_idx + 1} failed for {item_id}: {batch_error}. "
                        f"Falling back to individual processing."
                    )
                # Fallback: process individually
                for chunk in batch:
                    try:
                        embedding = self.generate_embedding(chunk)
                        chunk['embedding'] = embedding
                        processed_chunks.append(chunk)
                    except Exception as chunk_error:
                        logger.error(
                            f"Failed to process chunk {chunk.get('chunk_index')} "
                            f"for {item_id}: {chunk_error}"
                        )
                        # Skip this chunk, continue with others
        
        return processed_chunks
    
    def store_in_bigquery(
        self, item_id: str, feed: str, metadata: Dict, chunks: List[Dict]
    ):
        """Store item and chunks in BigQuery."""
        # Store item
        items_table = self.bq_client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{ITEMS_TABLE}")
        item_row = {
            'item_id': item_id,
            'feed': feed,
            'source_uri': metadata.get('source', 'NVIDIA Developer Blog'),
            'title': metadata.get('title', ''),
            'publication_date': metadata.get('publication_date'),
        }
        self.bq_client.insert_rows_json(items_table, [item_row])
        
        # Store chunks
        chunks_table = self.bq_client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{CHUNKS_TABLE}")
        chunk_rows = []
        for chunk in chunks:
            chunk_row = {
                'chunk_id': f"{item_id}_chunk_{chunk['chunk_index']}",
                'item_id': item_id,
                'feed': feed,
                'source_uri': metadata.get('source', 'NVIDIA Developer Blog'),
                'text': chunk['text'],
                'chunk_index': chunk['chunk_index'],
                'publication_date': metadata.get('publication_date'),
                'title': metadata.get('title', ''),
                'embedding': chunk['embedding']
            }
            chunk_rows.append(chunk_row)
        
        self.bq_client.insert_rows_json(chunks_table, chunk_rows)
        logger.info(f"Stored {len(chunk_rows)} chunks for {item_id}")
    
    def process_item_with_retry(
        self, item_id: str, feed: str, content: str, metadata: Dict
    ) -> bool:
        """
        Process item with automatic retry using progressively smaller chunks.
        
        If token limit errors occur, automatically retries with smaller chunk sizes.
        Never gives up - keeps trying until it works or reaches minimum chunk size.
        """
        content_size = len(content.encode('utf-8'))
        
        # Progressive chunk size reduction strategy
        # Start with adaptive size, then reduce if needed
        chunk_size_strategies = [
            self.content_handler.determine_chunking_config(content_size),
            ChunkingConfig(chunk_size_tokens=500, overlap_tokens=100),  # Smaller
            ChunkingConfig(chunk_size_tokens=300, overlap_tokens=60),    # Even smaller
            ChunkingConfig(chunk_size_tokens=200, overlap_tokens=40),    # Very small
            ChunkingConfig(chunk_size_tokens=100, overlap_tokens=20),    # Minimum
        ]
        
        last_error = None
        
        for attempt, config in enumerate(chunk_size_strategies, 1):
            try:
                logger.info(
                    f"Attempt {attempt} for {item_id}: "
                    f"using {config.chunk_size_tokens}-token chunks"
                )
                
                # Chunk content with current strategy
                chunks = self.chunk_content(
                    content,
                    chunk_size=config.chunk_size_tokens,
                    overlap=config.overlap_tokens
                )
                logger.info(f"Chunked into {len(chunks)} chunks")
                
                # Create batches that respect token limits
                batches = self.content_handler.create_embedding_batches(chunks, config)
                logger.info(f"Grouped into {len(batches)} batches")
                
                # Process batches with fallback
                processed_chunks = self.process_batches_with_fallback(item_id, batches)
                
                if not processed_chunks:
                    # If no chunks processed, try next smaller size
                    logger.warning(
                        f"No chunks processed for {item_id} with {config.chunk_size_tokens}-token chunks. "
                        f"Trying smaller chunk size..."
                    )
                    continue
                
                # Success! Store in BigQuery
                self.store_in_bigquery(item_id, feed, metadata, processed_chunks)
                
                logger.info(
                    f"Successfully processed {item_id} on attempt {attempt}: "
                    f"{len(processed_chunks)}/{len(chunks)} chunks "
                    f"(used {config.chunk_size_tokens}-token chunks)"
                )
                return True
                
            except Exception as e:
                last_error = e
                is_token_error = self.is_token_limit_error(e)
                
                if is_token_error and attempt < len(chunk_size_strategies):
                    # Token limit error - try smaller chunks
                    logger.warning(
                        f"Token limit error on attempt {attempt} for {item_id}. "
                        f"Retrying with smaller chunk size ({chunk_size_strategies[attempt].chunk_size_tokens} tokens)..."
                    )
                    continue
                else:
                    # Other error or last attempt - log and return
                    logger.error(
                        f"Failed to process {item_id} on attempt {attempt}: {e}",
                        exc_info=True
                    )
                    if attempt == len(chunk_size_strategies):
                        # Last attempt failed
                        logger.error(
                            f"All retry attempts exhausted for {item_id}. "
                            f"Content may be too large or have other issues."
                        )
                    break
        
        # All attempts failed
        self.results['errors'].append({
            'item_id': item_id,
            'feed': feed,
            'error': type(last_error).__name__ if last_error else 'Unknown',
            'message': str(last_error) if last_error else 'All retry attempts failed'
        })
        return False
    
    def process_item(self, item_id: str, feed: str) -> bool:
        """Process a single item with full error handling and automatic retry."""
        try:
            logger.info(f"Processing item: {item_id} (feed: {feed})")
            
            # Load content
            content = self.load_item_content(item_id, feed)
            if not content:
                logger.warning(f"Could not load content for {item_id}")
                return False
            
            # Parse metadata
            metadata = self.parse_item_metadata(content)
            content_size = len(content.encode('utf-8'))
            
            logger.info(
                f"Content size: {content_size / 1024:.1f}KB"
            )
            
            # Process with automatic retry
            return self.process_item_with_retry(item_id, feed, content, metadata)
            
        except Exception as e:
            logger.error(
                f"Failed to process {item_id}: {e}",
                exc_info=True
            )
            self.results['errors'].append({
                'item_id': item_id,
                'feed': feed,
                'error': type(e).__name__,
                'message': str(e)
            })
            return False
    
    def run(self):
        """Run the sync job with resilient error handling."""
        logger.info("Starting BigQuery incremental sync job")
        
        # Process each feed
        for feed in ['dev', 'official']:
            logger.info(f"Processing feed: {feed}")
            
            items = self.get_items_to_process(feed)
            self.results['total'] += len(items)
            
            if not items:
                logger.info(f"No new items for feed: {feed}")
                continue
            
            for item_id in items:
                if self.process_item(item_id, feed):
                    self.results['success'] += 1
                else:
                    self.results['failed'] += 1
        
        # Log summary
        logger.info(
            f"Incremental sync job completed: "
            f"{self.results['success']}/{self.results['total']} succeeded, "
            f"{self.results['failed']} failed"
        )
        
        if self.results['errors']:
            logger.warning(f"Errors encountered: {len(self.results['errors'])} items")
            for error in self.results['errors'][:5]:  # Log first 5 errors
                logger.warning(f"  - {error['item_id']}: {error['error']}")
        
        # Exit code: 0 if any items succeeded, 1 only if complete failure
        if self.results['success'] == 0 and self.results['total'] > 0:
            logger.error("CRITICAL: No items processed successfully")
            return 1
        elif self.results['total'] == 0:
            logger.info("No new items to process - job completed successfully")
            return 0
        else:
            logger.info(f"Job completed with partial/full success")
            return 0


def main():
    """Main entry point."""
    try:
        sync = ResilientBigQuerySync()
        exit_code = sync.run()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"CRITICAL: Job failed with exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
