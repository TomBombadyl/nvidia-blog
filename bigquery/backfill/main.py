#!/usr/bin/env python3
"""
Cloud Run Job: BigQuery Backfill
One-time script to populate BigQuery with all existing cleaned data from GCS.
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import List, Dict

# Try to import Cloud Logging, but fall back gracefully if not available
try:
    from google.cloud import logging as cloud_logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False

from backfill.config import (
    BUCKET_NAME,
    RSS_FEEDS,
    LOG_LEVEL,
    BATCH_SIZE
)
from backfill.gcs_reader import GCSReader
from backfill.processor import BackfillProcessor
from shared.metadata_extractor import extract_metadata_from_text, extract_source_uri_from_item_id

# Configure structured JSON logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Use Cloud Logging if available (in Cloud Run)
if CLOUD_LOGGING_AVAILABLE:
    try:
        client = cloud_logging.Client()
        handler = CloudLoggingHandler(client)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)
    except Exception:
        pass  # Fallback to stdout logging

logger = logging.getLogger(__name__)


def log_structured(level: str, message: str, **kwargs):
    """Emit structured JSON log entry."""
    log_entry = {
        "severity": level.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs
    }
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(log_entry))


def main():
    """Main entry point for backfill job."""
    log_structured(
        "info",
        "Starting BigQuery backfill job",
        bucket=BUCKET_NAME,
        feeds=list(RSS_FEEDS.keys())
    )
    
    try:
        # Initialize components
        gcs_reader = GCSReader(bucket_name=BUCKET_NAME)
        processor = BackfillProcessor()
        
        # Get all cleaned files
        log_structured("info", "Listing all cleaned files from GCS")
        all_files = gcs_reader.get_all_cleaned_files()
        log_structured("info", f"Found {len(all_files)} cleaned files", total_files=len(all_files))
        
        if not all_files:
            log_structured("warning", "No cleaned files found in GCS")
            return
        
        # Group by feed for processing
        files_by_feed = {}
        for file_info in all_files:
            feed = file_info["feed"]
            if feed not in files_by_feed:
                files_by_feed[feed] = []
            files_by_feed[feed].append(file_info)
        
        # Process each feed
        total_processed = 0
        total_errors = 0
        
        for feed, file_infos in files_by_feed.items():
            log_structured("info", f"Processing feed: {feed}", feed=feed, file_count=len(file_infos))
            
            # Extract item IDs
            item_ids = [f["item_id"] for f in file_infos]
            
            # Skip already processed items
            new_item_ids = processor.skip_already_processed(item_ids, feed)
            log_structured(
                "info",
                f"Feed {feed}: {len(new_item_ids)} new items to process",
                feed=feed,
                new_count=len(new_item_ids),
                total_count=len(item_ids)
            )
            
            # Filter file_infos to only new items
            new_files = {f["item_id"]: f for f in file_infos if f["item_id"] in new_item_ids}
            
            # Process in batches
            batch_count = 0
            for i in range(0, len(new_item_ids), BATCH_SIZE):
                batch = new_item_ids[i:i + BATCH_SIZE]
                batch_count += 1
                
                log_structured(
                    "info",
                    f"Processing batch {batch_count}",
                    feed=feed,
                    batch_size=len(batch),
                    batch_number=batch_count
                )
                
                # Read cleaned text for batch
                items_to_process = []
                for item_id in batch:
                    file_info = new_files[item_id]
                    try:
                        cleaned_text = gcs_reader.read_cleaned_text(file_info["blob_path"])
                        
                        # Extract metadata from cleaned text header
                        metadata = extract_metadata_from_text(cleaned_text)
                        
                        # Try to get source URI from item_id if not in metadata
                        source_uri = metadata.get("source_uri") or extract_source_uri_from_item_id(item_id, feed)
                        
                        items_to_process.append({
                            "item_id": item_id,
                            "feed": feed,
                            "cleaned_text": cleaned_text,
                            "source_uri": source_uri,
                            "title": metadata.get("title"),
                            "publication_date": metadata.get("publication_date")
                        })
                    except Exception as e:
                        log_structured(
                            "error",
                            f"Error reading file for {item_id}",
                            item_id=item_id,
                            error=str(e)
                        )
                        total_errors += 1
                        continue
                
                # Process batch
                results = processor.process_batch(items_to_process)
                
                # Count successes and errors
                for result in results:
                    if result.get("success"):
                        total_processed += 1
                    else:
                        total_errors += 1
                        log_structured(
                            "error",
                            f"Failed to process item",
                            item_id=result.get("item_id"),
                            error=result.get("error")
                        )
                
                # Small delay between batches
                if i + BATCH_SIZE < len(new_item_ids):
                    time.sleep(1)
        
        log_structured(
            "info",
            "Backfill job completed",
            total_processed=total_processed,
            total_errors=total_errors
        )
        
        sys.exit(0 if total_errors == 0 else 1)
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        log_structured(
            "error",
            "Backfill job failed",
            error=str(e),
            error_type=type(e).__name__,
            traceback=error_traceback
        )
        logger.error("Full error traceback:\n%s", error_traceback)
        sys.exit(1)


if __name__ == "__main__":
    main()

