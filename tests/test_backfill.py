"""
Tests for BigQuery backfill script.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bigquery.backfill.gcs_reader import GCSReader
from bigquery.backfill.processor import BackfillProcessor


class TestGCSReader(unittest.TestCase):
    """Tests for GCS reader."""
    
    def test_extract_metadata_from_path(self):
        """Test extracting metadata from blob path."""
        reader = GCSReader()
        
        # Test dev feed
        metadata = reader.extract_metadata_from_path("dev/clean/item_123.txt")
        self.assertEqual(metadata["feed"], "dev")
        self.assertEqual(metadata["item_id"], "item_123")
        
        # Test official feed
        metadata = reader.extract_metadata_from_path("official/clean/item_456.txt")
        self.assertEqual(metadata["feed"], "official")
        self.assertEqual(metadata["item_id"], "item_456")


class TestBackfillProcessor(unittest.TestCase):
    """Tests for backfill processor."""
    
    @patch('bigquery.backfill.processor.EmbeddingClient')
    @patch('bigquery.backfill.processor.BigQueryClient')
    def test_process_item(self, mock_bq_client, mock_embedding_client):
        """Test processing a single item."""
        # Mock embedding client
        mock_embedding = Mock()
        mock_embedding.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
        mock_embedding_client.return_value = mock_embedding
        
        # Mock BigQuery client
        mock_bq = Mock()
        mock_bq_client.return_value = mock_bq
        
        processor = BackfillProcessor()
        
        # Test processing
        result = processor.process_item(
            item_id="test_item",
            feed="dev",
            cleaned_text="This is a test article with enough words to chunk properly. " * 100
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["item_id"], "test_item")
        self.assertGreater(result["chunks_count"], 0)


if __name__ == '__main__':
    unittest.main()

