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
    
    def get_items_to_process(self, feed: str) -> List[str]:
        """Get list of item IDs that need processing."""
        bucket = self.storage_client.bucket(BUCKET_NAME)
        prefix = f"{feed}/clean/"
        
        # Get all cleaned files
        blobs = bucket.list_blobs(prefix=prefix)
        item_ids = []
        for blob in blobs:
            if blob.name.endswith('.txt'):
                # Extract item_id from filename
                # Format: feed/clean/https___developer.nvidia.com_blog__p_XXXXX.txt
                filename = blob.name.split('/')[-1]
                item_id = filename.replace('.txt', '')
                item_ids.append(item_id)
        
        # Filter out already processed items
        processed_ids = self.get_processed_item_ids(feed)
        new_items = [item_id for item_id in item_ids if item_id not in processed_ids]
        
        logger.info(f"Feed {feed}: {len(new_items)} new items out of {len(item_ids)} total")
        return new_items
    
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
    
    def process_item(self, item_id: str, feed: str) -> bool:
        """Process a single item with full error handling."""
        try:
            logger.info(f"Processing item: {item_id} (feed: {feed})")
            
            # Load content
            content = self.load_item_content(item_id, feed)
            if not content:
                logger.warning(f"Could not load content for {item_id}")
                return False
            
            # Parse metadata
            metadata = self.parse_item_metadata(content)
            
            # Determine chunking strategy based on content size
            content_size = len(content.encode('utf-8'))
            config = self.content_handler.determine_chunking_config(content_size)
            
            logger.info(
                f"Content size: {content_size / 1024:.1f}KB, "
                f"using {config.chunk_size_tokens}-token chunks"
            )
            
            # Chunk content
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
                logger.warning(f"No chunks processed for {item_id}")
                return False
            
            # Store in BigQuery
            self.store_in_bigquery(item_id, feed, metadata, processed_chunks)
            
            logger.info(
                f"Successfully processed {item_id}: "
                f"{len(processed_chunks)}/{len(chunks)} chunks"
            )
            return True
            
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
