#!/usr/bin/env python3
"""
Validation script to compare BigQuery vs RAG Corpus results.
"""

import json
import logging
import sys
import os
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bigquery.mcp_server.query_bigquery import BigQueryRAGQuery
from bigquery.mcp_server.config import (
    PROJECT_ID,
    BIGQUERY_DATASET,
    BIGQUERY_TABLE_CHUNKS
)

# Import RAG query for comparison
from mcp.query_rag import RAGQuery
from mcp.config import RAG_CORPUS, REGION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_queries(test_queries: List[str]) -> Dict:
    """
    Compare results from BigQuery and RAG Corpus for the same queries.
    
    Args:
        test_queries: List of test query strings
        
    Returns:
        Dictionary with comparison results
    """
    # Initialize both query interfaces
    logger.info("Initializing BigQuery query interface...")
    bigquery_query = BigQueryRAGQuery(
        PROJECT_ID,
        BIGQUERY_DATASET,
        BIGQUERY_TABLE_CHUNKS,
        enable_transformation=True,
        enable_grading=True,
        max_refinement_iterations=2
    )
    
    logger.info("Initializing RAG Corpus query interface...")
    rag_query = RAGQuery(
        RAG_CORPUS,
        REGION,
        enable_transformation=True,
        enable_grading=True,
        max_refinement_iterations=2
    )
    
    comparison_results = {
        "queries": [],
        "summary": {
            "total_queries": len(test_queries),
            "matching_results": 0,
            "different_results": 0,
            "bigquery_only": 0,
            "rag_only": 0
        }
    }
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        # Query BigQuery
        try:
            bq_result = bigquery_query.query(
                query_text=query,
                similarity_top_k=10,
                vector_distance_threshold=0.7
            )
            bq_count = bq_result.get("count", 0)
            bq_contexts = bq_result.get("contexts", [])
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            bq_count = 0
            bq_contexts = []
        
        # Query RAG Corpus
        try:
            rag_result = rag_query.query(
                query_text=query,
                similarity_top_k=10,
                vector_distance_threshold=0.7
            )
            rag_count = rag_result.get("count", 0)
            rag_contexts = rag_result.get("contexts", [])
        except Exception as e:
            logger.error(f"RAG Corpus query failed: {e}")
            rag_count = 0
            rag_contexts = []
        
        # Compare results
        query_comparison = {
            "query": query,
            "bigquery_count": bq_count,
            "rag_count": rag_count,
            "count_match": bq_count == rag_count,
            "bigquery_contexts": len(bq_contexts),
            "rag_contexts": len(rag_contexts)
        }
        
        if bq_count == rag_count:
            comparison_results["summary"]["matching_results"] += 1
        elif bq_count > 0 and rag_count == 0:
            comparison_results["summary"]["bigquery_only"] += 1
        elif bq_count == 0 and rag_count > 0:
            comparison_results["summary"]["rag_only"] += 1
        else:
            comparison_results["summary"]["different_results"] += 1
        
        comparison_results["queries"].append(query_comparison)
        
        logger.info(
            f"Query: {query[:50]}... | "
            f"BigQuery: {bq_count} results | "
            f"RAG: {rag_count} results"
        )
    
    return comparison_results


def main():
    """Main entry point."""
    # Test queries
    test_queries = [
        "CUDA programming best practices",
        "TensorRT inference optimization",
        "GPU memory management",
        "Multi-GPU training",
        "Deep learning frameworks"
    ]
    
    logger.info("Starting validation comparison...")
    results = compare_queries(test_queries)
    
    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total queries: {results['summary']['total_queries']}")
    print(f"Matching results: {results['summary']['matching_results']}")
    print(f"Different results: {results['summary']['different_results']}")
    print(f"BigQuery only: {results['summary']['bigquery_only']}")
    print(f"RAG only: {results['summary']['rag_only']}")
    print("=" * 50)
    
    # Save results to file
    output_file = "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

