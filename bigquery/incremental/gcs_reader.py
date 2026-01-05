"""
GCS Reader for Incremental Sync
Reads processed_ids.json and cleaned text files from GCS.
"""

import logging
from typing import List, Set, Optional
from google.cloud import storage
import json
from incremental.config import BUCKET_NAME, RSS_FEEDS

logger = logging.getLogger(__name__)


class GCSReader:
    """Reads processed IDs and cleaned text files from GCS."""
    
    def __init__(self, bucket_name: str = BUCKET_NAME):
        """
        Initialize GCS reader.
        
        Args:
            bucket_name: GCS bucket name
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        logger.info(f"Initialized GCS reader for bucket: {bucket_name}")
    
    def read_processed_ids(self, feed: str) -> Set[str]:
        """
        Read processed item IDs from processed_ids.json.
        
        Args:
            feed: Feed name (dev or official)
            
        Returns:
            Set of processed item IDs
        """
        try:
            folder = RSS_FEEDS[feed]["folder"]
            blob_path = f"{folder}/processed_ids.json"
            
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                logger.warning(f"processed_ids.json not found for feed {feed}")
                return set()
            
            content = blob.download_as_text()
            data = json.loads(content)
            
            # Extract IDs from the JSON structure
            item_ids = set(data.get("ids", []))
            
            logger.info(f"Read {len(item_ids)} processed IDs for feed: {feed}")
            return item_ids
            
        except Exception as e:
            logger.error(f"Error reading processed_ids.json for feed {feed}: {e}")
            return set()
    
    def read_cleaned_text(self, feed: str, item_id: str) -> Optional[str]:
        """
        Read cleaned text file for an item.
        
        Args:
            feed: Feed name (dev or official)
            item_id: Item ID
            
        Returns:
            Text content as string, or None if file doesn't exist
        """
        try:
            folder = RSS_FEEDS[feed]["folder"]
            blob_path = f"{folder}/clean/{item_id}.txt"
            
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                logger.warning(f"Cleaned text file not found: {blob_path}")
                return None
            
            text = blob.download_as_text()
            logger.debug(f"Read {len(text)} characters from {blob_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error reading cleaned text for {item_id}: {e}")
            return None
    
    def get_new_item_ids(self, feed: str, processed_ids_in_bq: Set[str]) -> List[str]:
        """
        Get list of new item IDs that need processing.
        
        Args:
            feed: Feed name
            processed_ids_in_bq: Set of item IDs already in BigQuery
            
        Returns:
            List of new item IDs to process
        """
        # Read processed IDs from GCS
        gcs_processed_ids = self.read_processed_ids(feed)
        
        # Find difference: items in GCS but not in BigQuery
        new_ids = list(gcs_processed_ids - processed_ids_in_bq)
        
        logger.info(
            f"Feed {feed}: {len(new_ids)} new items out of "
            f"{len(gcs_processed_ids)} total processed"
        )
        
        return new_ids

