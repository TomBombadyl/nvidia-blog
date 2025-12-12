"""
Enhanced RAG Corpus Query Module with Query Transformation and Answer Grading
Provides query capabilities for Vertex AI RAG Corpus with iterative refinement.
"""

import logging
import re
import requests
from typing import Dict, List
from google.auth import default
from google.auth.transport.requests import Request
from rag_query_transformer import QueryTransformer
from rag_answer_grader import AnswerGrader
from config import GEMINI_MODEL_LOCATION, GEMINI_MODEL_NAME, RAG_VECTOR_DISTANCE_THRESHOLD

logger = logging.getLogger(__name__)
# Enable debug logging to diagnose API response structure
logger.setLevel(logging.DEBUG)


class RAGQuery:
    """Enhanced RAG query interface with query transformation and answer grading."""

    def __init__(
        self,
        rag_corpus_name: str,
        region: str,
        enable_transformation: bool = True,
        enable_grading: bool = True,
        max_refinement_iterations: int = 2
    ):
        """
        Initialize enhanced RAG query interface.
        
        Args:
            rag_corpus_name: Full RAG corpus resource name
                e.g., projects/PROJECT/locations/REGION/ragCorpora/CORPUS_ID
            region: GCP region
            enable_transformation: Enable query transformation (default: True)
            enable_grading: Enable answer grading and refinement (default: True)
            max_refinement_iterations: Maximum refinement iterations (default: 2)
        """
        self.rag_corpus_name = rag_corpus_name
        self.region = region
        self.enable_transformation = enable_transformation
        self.enable_grading = enable_grading
        self.max_refinement_iterations = max_refinement_iterations
        
        # Extract project ID from rag_corpus_name
        # Format: projects/PROJECT/locations/REGION/ragCorpora/CORPUS_ID
        parts = rag_corpus_name.split('/')
        self.project_id = parts[1]

        # Get credentials for API calls
        self.credentials, _ = default()

        # Base URL for RAG API
        self.base_url = (
            f"https://{region}-aiplatform.googleapis.com/v1beta1"
        )
        
        # Initialize query transformer and grader if enabled
        if self.enable_transformation or self.enable_grading:
            if self.enable_transformation:
                self.query_transformer = QueryTransformer(
                    project_id=self.project_id,
                    region=region,
                    model_name=GEMINI_MODEL_NAME,
                    gemini_location=GEMINI_MODEL_LOCATION
                )
            else:
                self.query_transformer = None
            
            if self.enable_grading:
                self.answer_grader = AnswerGrader(
                    project_id=self.project_id,
                    region=region,
                    model_name=GEMINI_MODEL_NAME,
                    gemini_location=GEMINI_MODEL_LOCATION
                )
            else:
                self.answer_grader = None
        else:
            self.query_transformer = None
            self.answer_grader = None

        logger.info(
            f"Initialized enhanced RAG query interface for corpus: {rag_corpus_name} "
            f"(transformation={enable_transformation}, grading={enable_grading})"
        )
    
    def _get_access_token(self) -> str:
        """Get access token for API calls."""
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self.credentials.token
    
    def _retrieve_contexts(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = RAG_VECTOR_DISTANCE_THRESHOLD
    ) -> List[Dict]:
        """
        Internal method to retrieve contexts from RAG Corpus.
        
        Args:
            query_text: The query text to search for
            similarity_top_k: Number of top results
            vector_distance_threshold: Minimum similarity threshold
        
        Returns:
            List of context dictionaries
        """
        # Retrieve contexts endpoint
        retrieve_url = (
            f"{self.base_url}/projects/{self.project_id}/"
            f"locations/{self.region}:retrieveContexts"
        )
        
        # Prepare request body
        request_body = {
            "vertex_rag_store": {
                "rag_resources": {
                    "rag_corpus": self.rag_corpus_name
                },
                "vector_distance_threshold": vector_distance_threshold
            },
            "query": {
                "text": query_text,
                "similarity_top_k": similarity_top_k
            }
        }

        # Make API call
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            retrieve_url,
            headers=headers,
            json=request_body,
            timeout=60
        )

        if response.status_code != 200:
            error_msg = (
                f"Failed to query RAG Corpus: {response.status_code} - "
                f"{response.text}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        result = response.json()
        
        # Debug: Log full API response structure to diagnose empty text issue
        if logger.isEnabledFor(logging.DEBUG):
            import json
            logger.debug(f"=== RAG API RESPONSE DEBUG ===")
            logger.debug(f"Raw API response keys: {list(result.keys())}")
            logger.debug(f"Full API response: {json.dumps(result, indent=2, default=str)[:2000]}")  # First 2000 chars
            if "contexts" in result:
                logger.debug(f"Contexts type: {type(result.get('contexts'))}")
                if isinstance(result.get("contexts"), dict):
                    logger.debug(f"Contexts dict keys: {list(result.get('contexts', {}).keys())}")
                    contexts_data = result.get("contexts", {})
                    if "contexts" in contexts_data:
                        logger.debug(f"Number of contexts: {len(contexts_data.get('contexts', []))}")
                        # Log first 3 contexts with full structure
                        for i, ctx in enumerate(contexts_data.get("contexts", [])[:3]):
                            logger.debug(f"Context {i+1} full structure: {json.dumps(ctx, indent=2, default=str)}")
            logger.debug(f"=== END RAG API RESPONSE DEBUG ===")
        
        # Extract contexts from response - handle multiple possible response structures
        # Try the expected structure first: contexts.contexts[]
        contexts = result.get("contexts", {}).get("contexts", [])
        
        # Fallback: if contexts is directly a list
        if not contexts and isinstance(result.get("contexts"), list):
            contexts = result.get("contexts", [])
        
        # Handle case where contexts might be wrapped in chunk objects
        # Check if contexts exist and are wrapped in chunk objects
        if contexts and isinstance(contexts, list) and len(contexts) > 0:
            # Check if first context has a 'chunk' key (API might wrap content)
            first_ctx = contexts[0]
            if isinstance(first_ctx, dict) and "chunk" in first_ctx:
                contexts = [ctx.get("chunk", ctx) for ctx in contexts]
            
            # Additional extraction: Check for nested text fields
            # Some API responses might have text under different paths
            normalized_contexts = []
            for idx, ctx in enumerate(contexts):
                if isinstance(ctx, dict):
                    # Try multiple possible text field locations with detailed logging
                    text = None
                    extraction_attempts = []
                    
                    # Attempt 1: Direct 'text' field
                    if ctx.get("text"):
                        text = ctx.get("text")
                        extraction_attempts.append("text")
                    # Attempt 2: Direct 'content' field
                    elif ctx.get("content"):
                        text = ctx.get("content")
                        extraction_attempts.append("content")
                    # Attempt 3: Nested chunk.text
                    elif isinstance(ctx.get("chunk"), dict) and ctx.get("chunk", {}).get("text"):
                        text = ctx.get("chunk", {}).get("text")
                        extraction_attempts.append("chunk.text")
                    # Attempt 4: Nested chunk.content
                    elif isinstance(ctx.get("chunk"), dict) and ctx.get("chunk", {}).get("content"):
                        text = ctx.get("chunk", {}).get("content")
                        extraction_attempts.append("chunk.content")
                    # Attempt 5: chunk_text field
                    elif ctx.get("chunk_text"):
                        text = ctx.get("chunk_text")
                        extraction_attempts.append("chunk_text")
                    # Attempt 6: text_content field
                    elif ctx.get("text_content"):
                        text = ctx.get("text_content")
                        extraction_attempts.append("text_content")
                    else:
                        text = ""
                        extraction_attempts.append("NONE - all attempts failed")
                    
                    if logger.isEnabledFor(logging.DEBUG) and idx < 3:
                        logger.debug(f"Context {idx+1} text extraction: {extraction_attempts[-1]}, text_length={len(text) if text else 0}, available_keys={list(ctx.keys())}")
                    
                    # Create normalized context with text field
                    normalized_ctx = {
                        "text": text,
                        "distance": ctx.get("distance"),
                        "source_uri": ctx.get("source_uri") or ctx.get("uri"),
                    }
                    # Preserve any other fields
                    for key, value in ctx.items():
                        if key not in normalized_ctx:
                            normalized_ctx[key] = value
                    normalized_contexts.append(normalized_ctx)
                else:
                    normalized_contexts.append(ctx)
            contexts = normalized_contexts
            
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
            
            # Debug: Log first context structure to see where text might be
            if logger.isEnabledFor(logging.DEBUG) and contexts:
                logger.debug(f"First context keys: {list(contexts[0].keys()) if isinstance(contexts[0], dict) else 'Not a dict'}")
                logger.debug(f"First context text length: {len(contexts[0].get('text', '')) if isinstance(contexts[0], dict) else 'N/A'}")
        
        # Log warning if contexts found but text fields are empty
        if contexts:
            empty_count = sum(1 for ctx in contexts if isinstance(ctx, dict) and not ctx.get("text") and not ctx.get("content"))
            if empty_count > 0:
                # Log full structure of first empty context for debugging
                first_empty = next((ctx for ctx in contexts if isinstance(ctx, dict) and not ctx.get("text") and not ctx.get("content")), None)
                if first_empty:
                    logger.warning(
                        f"Found {len(contexts)} contexts but {empty_count} have empty text fields. "
                        f"Sample empty context keys: {list(first_empty.keys())}"
                    )
        
        # Log warning if no contexts found
        if not contexts:
            logger.warning(
                f"No contexts found in response. Response structure: {list(result.keys())}"
            )
        
        return contexts

    def query(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = RAG_VECTOR_DISTANCE_THRESHOLD
    ) -> Dict:
        """
        Enhanced query RAG Corpus with transformation, grading, and iterative refinement.

        Args:
            query_text: The original user query text
            similarity_top_k: Number of top results (default: 10)
            vector_distance_threshold: Minimum similarity (default: 0.5)

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
            # Step 1: Transform query if enabled
            if self.enable_transformation and self.query_transformer:
                logger.info(f"Transforming query: '{original_query[:50]}...'")
                current_query = self.query_transformer.transform_query(original_query)
                logger.info(f"Transformed to: '{current_query[:50]}...'")
            
            # Step 2: Iterative retrieval with grading and refinement
            best_contexts = []
            best_grade = None
            
            for iteration in range(self.max_refinement_iterations + 1):
                logger.info(
                    f"Retrieval iteration {iteration + 1}/{self.max_refinement_iterations + 1}: "
                    f"'{current_query[:50]}...'"
                )
                
                # Retrieve contexts
                contexts = self._retrieve_contexts(
                    query_text=current_query,
                    similarity_top_k=similarity_top_k,
                    vector_distance_threshold=vector_distance_threshold
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
            logger.error(f"Error querying RAG Corpus: {e}")
            raise
