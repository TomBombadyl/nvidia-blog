"""
GCS Reader for Backfill
Reads all cleaned text files from GCS bucket.
"""

import logging
from typing import List, Dict, Optional
from google.cloud import storage
from backfill.config import BUCKET_NAME, RSS_FEEDS

logger = logging.getLogger(__name__)


class GCSReader:
    """Reads cleaned text files from GCS."""
    
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
    
    def list_cleaned_files(self, feed: str) -> List[str]:
        """
        List all cleaned text files for a feed.
        
        Args:
            feed: Feed name (dev or official)
            
        Returns:
            List of blob paths to cleaned text files
        """
        try:
            folder = RSS_FEEDS[feed]["folder"]
            prefix = f"{folder}/clean/"
            
            blobs = self.bucket.list_blobs(prefix=prefix)
            file_paths = [blob.name for blob in blobs if blob.name.endswith('.txt')]
            
            logger.info(f"Found {len(file_paths)} cleaned files for feed: {feed}")
            return file_paths
            
        except Exception as e:
            logger.error(f"Error listing files for feed {feed}: {e}")
            raise
    
    def read_cleaned_text(self, blob_path: str) -> str:
        """
        Read cleaned text file from GCS.
        
        Args:
            blob_path: Path to cleaned text file (e.g., "dev/clean/item_id.txt")
            
        Returns:
            Text content as string
        """
        try:
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                raise FileNotFoundError(f"File not found: {blob_path}")
            
            text = blob.download_as_text()
            logger.debug(f"Read {len(text)} characters from {blob_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error reading file {blob_path}: {e}")
            raise
    
    def extract_metadata_from_path(self, blob_path: str) -> Dict:
        """
        Extract metadata from file path.
        
        Args:
            blob_path: Path like "dev/clean/item_id.txt" or "official/clean/item_id.txt"
            
        Returns:
            Dictionary with feed and item_id
        """
        # Extract feed and item_id from path
        # Format: {feed}/clean/{item_id}.txt
        parts = blob_path.split('/')
        if len(parts) < 3:
            raise ValueError(f"Invalid blob path format: {blob_path}")
        
        feed = parts[0]
        item_id = parts[2].replace('.txt', '')
        
        return {
            "feed": feed,
            "item_id": item_id,
            "blob_path": blob_path
        }
    
    def get_all_cleaned_files(self) -> List[Dict]:
        """
        Get all cleaned files from all feeds.
        
        Returns:
            List of dictionaries with feed, item_id, and blob_path
        """
        all_files = []
        
        for feed in RSS_FEEDS.keys():
            file_paths = self.list_cleaned_files(feed)
            
            for blob_path in file_paths:
                metadata = self.extract_metadata_from_path(blob_path)
                all_files.append(metadata)
        
        logger.info(f"Found {len(all_files)} total cleaned files across all feeds")
        return all_files

