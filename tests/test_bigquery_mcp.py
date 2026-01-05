"""
Tests for BigQuery MCP server.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestBigQueryRAGQuery(unittest.TestCase):
    """Tests for BigQuery RAG query."""
    
    @patch('bigquery.mcp_server.query_bigquery.bigquery.Client')
    @patch('bigquery.mcp_server.query_bigquery.TextEmbeddingModel')
    def test_generate_query_embedding(self, mock_model_class, mock_bq_client):
        """Test generating query embedding."""
        # Mock embedding model
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 768
        mock_model.get_embeddings.return_value = [mock_embedding]
        mock_model_class.from_pretrained.return_value = mock_model
        
        from bigquery.mcp_server.query_bigquery import BigQueryRAGQuery
        
        query = BigQueryRAGQuery(
            project_id="test-project",
            dataset_id="test_dataset",
            table_name="chunks",
            enable_transformation=False,
            enable_grading=False
        )
        
        embedding = query._generate_query_embedding("test query")
        
        self.assertEqual(len(embedding), 768)
        self.assertIsInstance(embedding, list)
    
    @patch('bigquery.mcp_server.query_bigquery.bigquery.Client')
    @patch('bigquery.mcp_server.query_bigquery.TextEmbeddingModel')
    def test_retrieve_contexts(self, mock_model_class, mock_bq_client_class):
        """Test retrieving contexts from BigQuery."""
        # Mock BigQuery client and query job
        mock_bq_client = Mock()
        mock_query_job = Mock()
        mock_result = Mock()
        
        # Mock result rows
        mock_row = Mock()
        mock_row.text = "Test context"
        mock_row.source_uri = "https://test.com"
        mock_row.distance = 0.5
        mock_row.chunk_id = "chunk1"
        mock_row.item_id = "item1"
        mock_row.feed = "dev"
        mock_row.chunk_index = 0
        mock_row.publication_date = None
        mock_row.title = "Test Title"
        
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_query_job.result.return_value = mock_result
        mock_bq_client.query.return_value = mock_query_job
        mock_bq_client_class.return_value = mock_bq_client
        
        # Mock embedding model
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 768
        mock_model.get_embeddings.return_value = [mock_embedding]
        mock_model_class.from_pretrained.return_value = mock_model
        
        from bigquery.mcp_server.query_bigquery import BigQueryRAGQuery
        
        query = BigQueryRAGQuery(
            project_id="test-project",
            dataset_id="test_dataset",
            table_name="chunks",
            enable_transformation=False,
            enable_grading=False
        )
        
        contexts = query._retrieve_contexts("test query", similarity_top_k=10)
        
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["text"], "Test context")
        self.assertEqual(contexts[0]["source_uri"], "https://test.com")


if __name__ == '__main__':
    unittest.main()

