"""
BigQuery Client Wrapper
Handles BigQuery operations with batch inserts, idempotent operations, and retry logic.
"""

import logging
import uuid
from typing import List, Dict, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for BigQuery operations."""
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        items_table: str = "items",
        chunks_table: str = "chunks"
    ):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            items_table: Items table name
            chunks_table: Chunks table name
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.items_table = items_table
        self.chunks_table = chunks_table
        
        self.client = bigquery.Client(project=project_id)
        self.items_table_ref = f"{project_id}.{dataset_id}.{items_table}"
        self.chunks_table_ref = f"{project_id}.{dataset_id}.{chunks_table}"
        
        logger.info(
            f"Initialized BigQuery client: "
            f"items={self.items_table_ref}, chunks={self.chunks_table_ref}"
        )
    
    @staticmethod
    def _generate_chunk_id(item_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{item_id}_chunk_{chunk_index}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def insert_item(self, item_data: Dict) -> None:
        """
        Insert or update item metadata (idempotent).
        
        Args:
            item_data: Dictionary with item fields (item_id, feed, source_uri, etc.)
        """
        try:
            # Use MERGE to handle duplicates idempotently
            merge_sql = f"""
            MERGE `{self.items_table_ref}` AS target
            USING (
                SELECT
                    @item_id AS item_id,
                    @feed AS feed,
                    @source_uri AS source_uri,
                    @title AS title,
                    @publication_date AS publication_date,
                    @processed_at AS processed_at
            ) AS source
            ON target.item_id = source.item_id AND target.feed = source.feed
            WHEN MATCHED THEN
                UPDATE SET
                    source_uri = source.source_uri,
                    title = source.title,
                    publication_date = source.publication_date,
                    processed_at = source.processed_at
            WHEN NOT MATCHED THEN
                INSERT (item_id, feed, source_uri, title, publication_date, processed_at)
                VALUES (source.item_id, source.feed, source.source_uri, source.title, 
                        source.publication_date, source.processed_at)
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("item_id", "STRING", item_data["item_id"]),
                    bigquery.ScalarQueryParameter("feed", "STRING", item_data["feed"]),
                    bigquery.ScalarQueryParameter("source_uri", "STRING", item_data.get("source_uri", "")),
                    bigquery.ScalarQueryParameter("title", "STRING", item_data.get("title")),
                    bigquery.ScalarQueryParameter(
                        "publication_date", 
                        "TIMESTAMP", 
                        item_data.get("publication_date")
                    ),
                    bigquery.ScalarQueryParameter(
                        "processed_at",
                        "TIMESTAMP",
                        item_data.get("processed_at")
                    )
                ]
            )
            
            query_job = self.client.query(merge_sql, job_config=job_config)
            query_job.result()  # Wait for completion
            
            logger.info(f"Inserted/updated item: {item_data['item_id']}")
            
        except Exception as e:
            logger.error(f"Error inserting item {item_data.get('item_id')}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def insert_chunks(self, chunks_data: List[Dict]) -> None:
        """
        Insert chunks with embeddings (idempotent, batch insert using MERGE).
        
        Args:
            chunks_data: List of dictionaries with chunk fields
        """
        if not chunks_data:
            return
        
        try:
            # Use MERGE for idempotent inserts/updates
            # Process in batches to avoid query size limits
            batch_size = 100
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i + batch_size]
                
                # Build MERGE statement with UNION ALL for batch
                values_parts = []
                query_parameters = []
                
                for idx, chunk in enumerate(batch):
                    param_prefix = f"p{idx}_"
                    values_parts.append(f"""
                        SELECT
                            @{param_prefix}chunk_id AS chunk_id,
                            @{param_prefix}item_id AS item_id,
                            @{param_prefix}feed AS feed,
                            @{param_prefix}source_uri AS source_uri,
                            @{param_prefix}text AS text,
                            @{param_prefix}embedding AS embedding,
                            @{param_prefix}chunk_index AS chunk_index,
                            @{param_prefix}chunk_size_words AS chunk_size_words,
                            @{param_prefix}publication_date AS publication_date,
                            @{param_prefix}title AS title
                    """)
                    
                    query_parameters.extend([
                        bigquery.ScalarQueryParameter(f"{param_prefix}chunk_id", "STRING", chunk["chunk_id"]),
                        bigquery.ScalarQueryParameter(f"{param_prefix}item_id", "STRING", chunk["item_id"]),
                        bigquery.ScalarQueryParameter(f"{param_prefix}feed", "STRING", chunk["feed"]),
                        bigquery.ScalarQueryParameter(f"{param_prefix}source_uri", "STRING", chunk["source_uri"]),
                        bigquery.ScalarQueryParameter(f"{param_prefix}text", "STRING", chunk["text"]),
                        bigquery.ArrayQueryParameter(f"{param_prefix}embedding", "FLOAT64", chunk["embedding"]),
                        bigquery.ScalarQueryParameter(f"{param_prefix}chunk_index", "INT64", chunk["chunk_index"]),
                        bigquery.ScalarQueryParameter(f"{param_prefix}chunk_size_words", "INT64", chunk.get("chunk_size_words")),
                        bigquery.ScalarQueryParameter(f"{param_prefix}publication_date", "TIMESTAMP", chunk.get("publication_date")),
                        bigquery.ScalarQueryParameter(f"{param_prefix}title", "STRING", chunk.get("title"))
                    ])
                
                merge_sql = f"""
                MERGE `{self.chunks_table_ref}` AS target
                USING (
                    {' UNION ALL '.join(values_parts)}
                ) AS source
                ON target.chunk_id = source.chunk_id
                WHEN MATCHED THEN
                    UPDATE SET
                        item_id = source.item_id,
                        feed = source.feed,
                        source_uri = source.source_uri,
                        text = source.text,
                        embedding = source.embedding,
                        chunk_index = source.chunk_index,
                        chunk_size_words = source.chunk_size_words,
                        publication_date = source.publication_date,
                        title = source.title
                WHEN NOT MATCHED THEN
                    INSERT (chunk_id, item_id, feed, source_uri, text, embedding, chunk_index, chunk_size_words, publication_date, title)
                    VALUES (source.chunk_id, source.item_id, source.feed, source.source_uri, source.text, 
                            source.embedding, source.chunk_index, source.chunk_size_words, source.publication_date, source.title)
                """
                
                job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
                query_job = self.client.query(merge_sql, job_config=job_config)
                query_job.result()  # Wait for completion
                
                logger.info(f"Inserted/updated {len(batch)} chunks (batch {i//batch_size + 1})")
            
            logger.info(f"Inserted/updated {len(chunks_data)} chunks total")
            
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            raise
    
    def get_processed_item_ids(self, feed: Optional[str] = None) -> set:
        """
        Get set of item IDs already in BigQuery.
        
        Args:
            feed: Optional feed name to filter by
            
        Returns:
            Set of item_id strings
        """
        try:
            if feed:
                query = f"""
                SELECT DISTINCT item_id
                FROM `{self.items_table_ref}`
                WHERE feed = @feed
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("feed", "STRING", feed)
                    ]
                )
            else:
                query = f"""
                SELECT DISTINCT item_id
                FROM `{self.items_table_ref}`
                """
                job_config = bigquery.QueryJobConfig()
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            item_ids = {row.item_id for row in results}
            logger.info(f"Found {len(item_ids)} processed item IDs in BigQuery")
            
            return item_ids
            
        except Exception as e:
            logger.error(f"Error getting processed item IDs: {e}")
            raise

