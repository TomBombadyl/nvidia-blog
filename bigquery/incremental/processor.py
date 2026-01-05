"""
Processor for Incremental Sync
Orchestrates reading new items, chunking, embedding, and writing to BigQuery.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from shared.text_chunker import chunk_text
from shared.embedding_client import EmbeddingClient
from shared.bigquery_client import BigQueryClient
from incremental.config import (
    PROJECT_ID,
    REGION,
    BIGQUERY_DATASET,
    BIGQUERY_TABLE_ITEMS,
    BIGQUERY_TABLE_CHUNKS,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE
)

logger = logging.getLogger(__name__)


class IncrementalProcessor:
    """Processes new cleaned text files and writes to BigQuery."""
    
    def __init__(self):
        """Initialize processor with clients."""
        self.embedding_client = EmbeddingClient(
            model_name=EMBEDDING_MODEL,
            project_id=PROJECT_ID,
            region=REGION
        )
        self.bq_client = BigQueryClient(
            project_id=PROJECT_ID,
            dataset_id=BIGQUERY_DATASET,
            items_table=BIGQUERY_TABLE_ITEMS,
            chunks_table=BIGQUERY_TABLE_CHUNKS
        )
        logger.info("Initialized incremental processor")
    
    def process_item(
        self,
        item_id: str,
        feed: str,
        cleaned_text: str,
        source_uri: Optional[str] = None,
        title: Optional[str] = None,
        publication_date: Optional[datetime] = None
    ) -> Dict:
        """
        Process a single item: chunk, embed, write to BigQuery.
        
        Args:
            item_id: Unique item identifier
            feed: Feed name (dev or official)
            cleaned_text: Cleaned text content
            source_uri: Optional source URI
            title: Optional title
            publication_date: Optional publication date
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing item: {item_id} (feed: {feed})")
            
            # Chunk text
            chunks = chunk_text(cleaned_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            logger.info(f"Chunked into {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_client.embed_batch(chunk_texts)
            
            # Prepare item metadata
            item_data = {
                "item_id": item_id,
                "feed": feed,
                "source_uri": source_uri or f"https://unknown/{item_id}",
                "title": title,
                "publication_date": publication_date,
                "processed_at": datetime.utcnow()
            }
            
            # Insert item
            self.bq_client.insert_item(item_data)
            
            # Prepare chunks data
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunk_id = BigQueryClient._generate_chunk_id(item_id, chunk['chunk_index'])
                
                chunk_data = {
                    "chunk_id": chunk_id,
                    "item_id": item_id,
                    "feed": feed,
                    "source_uri": source_uri or f"https://unknown/{item_id}",
                    "text": chunk['text'],
                    "embedding": embeddings[i],
                    "chunk_index": chunk['chunk_index'],
                    "chunk_size_words": chunk['word_count'],
                    "publication_date": publication_date,
                    "title": title
                }
                chunks_data.append(chunk_data)
            
            # Insert chunks
            self.bq_client.insert_chunks(chunks_data)
            
            logger.info(
                f"Successfully processed item {item_id}: "
                f"{len(chunks)} chunks inserted"
            )
            
            return {
                "item_id": item_id,
                "feed": feed,
                "chunks_count": len(chunks),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
            return {
                "item_id": item_id,
                "feed": feed,
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, items: List[Dict]) -> List[Dict]:
        """
        Process multiple items in batch.
        
        Args:
            items: List of item dictionaries with cleaned_text, item_id, feed, etc.
            
        Returns:
            List of processing results
        """
        results = []
        
        for item in items:
            result = self.process_item(
                item_id=item["item_id"],
                feed=item["feed"],
                cleaned_text=item["cleaned_text"],
                source_uri=item.get("source_uri"),
                title=item.get("title"),
                publication_date=item.get("publication_date")
            )
            results.append(result)
        
        return results

