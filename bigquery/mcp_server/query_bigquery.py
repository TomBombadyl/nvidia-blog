"""
BigQuery Query Module
Provides semantic search using BigQuery SQL queries with ML.DISTANCE.
Implements same interface as RAGQuery for compatibility.
"""

import logging
import re
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel

# Add parent directory to import transformer and grader from existing mcp/
parent_dir = os.path.join(os.path.dirname(__file__), '..', '..')
parent_dir = os.path.abspath(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from existing mcp/ directory using importlib with proper context
import importlib.util

# First, load config module and make it available for relative imports
config_path = os.path.join(parent_dir, 'archive', 'mcp-rag-corpus', 'config.py')
config_spec = importlib.util.spec_from_file_location("mcp.config", config_path)
config_module = importlib.util.module_from_spec(config_spec)
config_module.__package__ = 'mcp'
config_spec.loader.exec_module(config_module)
# Make 'config' available for relative imports
if 'config' not in sys.modules:
    sys.modules['config'] = config_module
sys.modules['mcp.config'] = config_module

# Import QueryTransformer
transformer_path = os.path.join(parent_dir, 'archive', 'mcp-rag-corpus', 'rag_query_transformer.py')
transformer_spec = importlib.util.spec_from_file_location("mcp.rag_query_transformer", transformer_path)
transformer_module = importlib.util.module_from_spec(transformer_spec)
transformer_module.__package__ = 'mcp'
transformer_module.__file__ = transformer_path
transformer_module.config = config_module
transformer_spec.loader.exec_module(transformer_module)
QueryTransformer = transformer_module.QueryTransformer

# Import AnswerGrader
grader_path = os.path.join(parent_dir, 'archive', 'mcp-rag-corpus', 'rag_answer_grader.py')
grader_spec = importlib.util.spec_from_file_location("mcp.rag_answer_grader", grader_path)
grader_module = importlib.util.module_from_spec(grader_spec)
grader_module.__package__ = 'mcp'
grader_module.__file__ = grader_path
grader_module.config = config_module
grader_spec.loader.exec_module(grader_module)
AnswerGrader = grader_module.AnswerGrader
from bigquery.mcp_server.config import (
    PROJECT_ID,
    REGION,
    BIGQUERY_DATASET,
    BIGQUERY_TABLE_CHUNKS,
    EMBEDDING_MODEL,
    RAG_VECTOR_DISTANCE_THRESHOLD,
    GEMINI_MODEL_LOCATION,
    GEMINI_MODEL_NAME
)
from bigquery.mcp_server.date_filter_extractor import DateFilterExtractor

logger = logging.getLogger(__name__)


class BigQueryRAGQuery:
    """BigQuery-based RAG query interface matching RAGQuery API."""
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_name: str,
        enable_transformation: bool = True,
        enable_grading: bool = True,
        max_refinement_iterations: int = 2
    ):
        """
        Initialize BigQuery RAG query interface.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_name: Chunks table name
            enable_transformation: Enable query transformation (default: True)
            enable_grading: Enable answer grading and refinement (default: True)
            max_refinement_iterations: Maximum refinement iterations (default: 2)
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.enable_transformation = enable_transformation
        self.enable_grading = enable_grading
        self.max_refinement_iterations = max_refinement_iterations
        
        self.client = bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset_id}.{table_name}"
        
        # Initialize Vertex AI for embeddings (models are in us-central1)
        import vertexai
        vertexai.init(project=project_id, location="us-central1")
        
        # Initialize embedding model for query embeddings
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        
        # Initialize transformer and grader (same as RAGQuery)
        if self.enable_transformation:
            self.query_transformer = QueryTransformer(
                project_id=project_id,
                region=REGION,
                model_name=GEMINI_MODEL_NAME,
                gemini_location=GEMINI_MODEL_LOCATION
            )
        else:
            self.query_transformer = None
        
        if self.enable_grading:
            self.answer_grader = AnswerGrader(
                project_id=project_id,
                region=REGION,
                model_name=GEMINI_MODEL_NAME,
                gemini_location=GEMINI_MODEL_LOCATION
            )
        else:
            self.answer_grader = None
        
        # Initialize date filter extractor
        self.date_filter_extractor = DateFilterExtractor()
        
        logger.info(
            f"Initialized BigQuery RAG query interface: {self.table_ref} "
            f"(transformation={enable_transformation}, grading={enable_grading})"
        )
    
    def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for query using Vertex AI."""
        try:
            embeddings = self.embedding_model.get_embeddings([query_text])
            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from model")
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def _retrieve_contexts(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = RAG_VECTOR_DISTANCE_THRESHOLD,
        date_filters: Optional[Dict[str, datetime]] = None
    ) -> List[Dict]:
        """Retrieve contexts using BigQuery SQL with ML.DISTANCE."""
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query_text)
            
            # Build WHERE clause conditions
            where_conditions = [
                "ML.DISTANCE(embedding, @query_embedding) < @distance_threshold"
            ]
            
            # Add date filters if present
            query_parameters = [
                bigquery.ArrayQueryParameter(
                    "query_embedding", "FLOAT64", query_embedding
                ),
                bigquery.ScalarQueryParameter(
                    "distance_threshold", "FLOAT64", vector_distance_threshold
                ),
                bigquery.ScalarQueryParameter(
                    "top_k", "INT64", similarity_top_k
                )
            ]
            
            if date_filters:
                start_date = date_filters.get('start_date')
                end_date = date_filters.get('end_date')
                
                if start_date and end_date:
                    where_conditions.append(
                        "publication_date >= @start_date AND publication_date < @end_date"
                    )
                    query_parameters.append(
                        bigquery.ScalarQueryParameter(
                            "start_date", "TIMESTAMP", start_date
                        )
                    )
                    query_parameters.append(
                        bigquery.ScalarQueryParameter(
                            "end_date", "TIMESTAMP", end_date
                        )
                    )
                    logger.info(
                        f"Applying date filter: {start_date} to {end_date}"
                    )
            
            # Build SQL query for similarity search using ML.DISTANCE
            # ML.DISTANCE computes cosine distance (lower is more similar)
            # Order by similarity first, then by publication_date DESC for recency
            # This matches the original RAG Corpus behavior but adds date sorting when available
            where_clause = " AND ".join(where_conditions)
            query_sql = f"""
            SELECT 
              chunk_id,
              item_id,
              feed,
              source_uri,
              text,
              chunk_index,
              publication_date,
              title,
              ML.DISTANCE(embedding, @query_embedding) AS distance
            FROM `{self.table_ref}`
            WHERE {where_clause}
            ORDER BY distance ASC, publication_date DESC NULLS LAST
            LIMIT @top_k
            """
            
            # Execute query with parameters
            job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
            
            query_job = self.client.query(query_sql, job_config=job_config)
            results = query_job.result()
            
            # Convert to list of dicts matching RAGQuery format
            contexts = []
            for row in results:
                contexts.append({
                    'text': row.text,
                    'source_uri': row.source_uri,
                    'distance': row.distance,
                    'chunk_id': row.chunk_id,
                    'item_id': row.item_id,
                    'feed': row.feed,
                    'chunk_index': row.chunk_index,
                    'publication_date': row.publication_date,
                    'title': row.title
                })
            
            # Apply same header filtering logic as original RAGQuery
            # Filter out header-only or very short chunks
            # These often occur when date queries match metadata headers
            filtered_contexts = []
            for ctx in contexts:
                if isinstance(ctx, dict):
                    text = ctx.get("text", "")
                    if text:
                        # Remove metadata headers and separators to check actual content
                        # Header pattern: "Publication Date: ...\nTitle: ...\nSource: ...\n\n---\n\n"
                        content_without_header = text
                        # Remove common header patterns
                        # Remove "Publication Date: ..." line
                        content_without_header = re.sub(
                            r'^Publication Date:.*?\n', '', content_without_header, flags=re.MULTILINE
                        )
                        # Remove "Title: ..." line
                        content_without_header = re.sub(
                            r'^Title:.*?\n', '', content_without_header, flags=re.MULTILINE
                        )
                        # Remove "Source: ..." line
                        content_without_header = re.sub(
                            r'^Source:.*?\n', '', content_without_header, flags=re.MULTILINE
                        )
                        # Remove separator lines
                        content_without_header = re.sub(
                            r'^---\s*$', '', content_without_header, flags=re.MULTILINE
                        )
                        # Strip whitespace
                        content_without_header = content_without_header.strip()
                        
                        # Keep chunk if it has substantial content after removing headers
                        # Minimum 100 characters of actual article content
                        if len(content_without_header) >= 100:
                            filtered_contexts.append(ctx)
                        else:
                            logger.debug(
                                f"Filtered out header-only chunk (content length after header removal: {len(content_without_header)})"
                            )
                    else:
                        # Skip chunks with empty text - don't add them to filtered contexts
                        logger.debug(f"Skipped chunk with empty text field (distance: {ctx.get('distance', 'N/A')})")
                else:
                    filtered_contexts.append(ctx)
            
            contexts = filtered_contexts
            
            # Log warning if contexts found but text fields are empty
            if contexts:
                empty_count = sum(1 for ctx in contexts if isinstance(ctx, dict) and not ctx.get("text") and not ctx.get("content"))
                if empty_count > 0:
                    logger.warning(
                        f"Found {len(contexts)} contexts but {empty_count} have empty text fields."
                    )
            
            # Log warning if no contexts found
            if not contexts:
                logger.warning("No contexts found after filtering")
            
            logger.info(f"Retrieved {len(contexts)} contexts from BigQuery (after filtering)")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving contexts from BigQuery: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = RAG_VECTOR_DISTANCE_THRESHOLD
    ) -> Dict:
        """
        Query BigQuery with same interface as RAGQuery.query().
        
        Args:
            query_text: The original user query text
            similarity_top_k: Number of top results (default: 10)
            vector_distance_threshold: Minimum similarity threshold (default: 0.7)
        
        Returns:
            Dictionary containing:
            - query: Original query text
            - transformed_query: Transformed query (if transformation enabled)
            - contexts: List of retrieved contexts
            - count: Number of contexts
            - grade: AnswerGrade object (if grading enabled)
            - refinement_iterations: Number of refinement iterations performed
        """
        original_query = query_text
        current_query = query_text
        refinement_iterations = 0
        
        try:
            # Step 0: Extract date filters BEFORE transformation
            # This allows us to filter by date even if query transformation changes the text
            date_filters = self.date_filter_extractor.extract_date_filters(original_query)
            if date_filters:
                logger.info(
                    f"Extracted date filters: {date_filters.get('start_date')} to "
                    f"{date_filters.get('end_date')}"
                )
            
            # Step 1: Transform query if enabled
            if self.enable_transformation and self.query_transformer:
                logger.info(f"Transforming query: '{original_query[:50]}...'")
                current_query = self.query_transformer.transform_query(original_query)
                logger.info(f"Transformed to: '{current_query[:50]}...'")
            
            # Step 2: Iterative retrieval with grading and refinement
            best_contexts = []
            best_grade = None
            
            # For date-only queries, lower distance threshold or remove it
            # to rely primarily on date filtering
            effective_threshold = vector_distance_threshold
            if date_filters and len(original_query.split()) <= 3:
                # Very short query with date filter - likely date-only query
                # Increase threshold to allow more results, rely on date filtering
                effective_threshold = min(vector_distance_threshold * 1.5, 0.95)
                logger.info(
                    f"Date-only query detected, adjusting threshold to {effective_threshold}"
                )
            
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(
                    f"Retrieval iteration {iteration + 1}/{self.max_refinement_iterations + 1}: "
                    f"'{current_query[:50]}...'"
                )
                
                # Retrieve contexts with date filters
                contexts = self._retrieve_contexts(
                    query_text=current_query,
                    similarity_top_k=similarity_top_k,
                    vector_distance_threshold=effective_threshold,
                    date_filters=date_filters
                )
                
                logger.info(f"Retrieved {len(contexts)} contexts")
                
                # Grade contexts if enabled
                if self.enable_grading and self.answer_grader and contexts:
                    grade = self.answer_grader.grade_contexts(
                        query=original_query,
                        contexts=contexts,
                        min_acceptable_score=0.6
                    )
                    
                    # Store best result so far
                    if best_grade is None or grade.score > best_grade.score:
                        best_contexts = contexts
                        best_grade = grade
                    
                    logger.info(
                        f"Iteration {iteration + 1} grade: score={grade.score:.2f}, "
                        f"relevance={grade.relevance:.2f}, should_refine={grade.should_refine}"
                    )
                    
                    # If grade is acceptable or max iterations reached, return best result
                    if not grade.should_refine or iteration >= self.max_refinement_iterations:
                        logger.info(
                            f"Stopping refinement: grade acceptable or max iterations reached"
                        )
                        break
                    
                    # Refine query for next iteration
                    if iteration < self.max_refinement_iterations:
                        logger.info("Refining query for next iteration...")
                        if self.query_transformer:
                            # Create refinement prompt
                            refinement_prompt = (
                                f"Original query: {original_query}\n"
                                f"Previous query: {current_query}\n"
                                f"Grade: {grade.reasoning}\n"
                                f"Improve the query to get better, more relevant results."
                            )
                            current_query = self.query_transformer.transform_query(refinement_prompt)
                            refinement_iterations += 1
                else:
                    # No grading enabled, return first result
                    best_contexts = contexts
                    break
            
            result = {
                "query": original_query,
                "contexts": best_contexts,
                "count": len(best_contexts),
                "refinement_iterations": refinement_iterations
            }
            
            # Add transformed query if transformation was used
            if self.enable_transformation and current_query != original_query:
                result["transformed_query"] = current_query
            
            # Add grade if grading was used
            if self.enable_grading and best_grade:
                result["grade"] = {
                    "score": best_grade.score,
                    "relevance": best_grade.relevance,
                    "completeness": best_grade.completeness,
                    "grounded": best_grade.grounded,
                    "reasoning": best_grade.reasoning,
                    "should_refine": best_grade.should_refine
                }
            
            logger.info(
                f"Query complete: '{original_query[:50]}...' → "
                f"{len(best_contexts)} contexts, {refinement_iterations} refinements"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying BigQuery: {e}")
            raise

