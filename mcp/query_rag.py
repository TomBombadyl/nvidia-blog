"""
Enhanced RAG Corpus Query Module with Query Transformation and Answer Grading
Provides query capabilities for Vertex AI RAG Corpus with iterative refinement.
"""

import logging
import requests
from typing import Dict, List
from google.auth import default
from google.auth.transport.requests import Request
from rag_query_transformer import QueryTransformer
from rag_answer_grader import AnswerGrader
from config import GEMINI_MODEL_LOCATION, GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)


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
        vector_distance_threshold: float = 0.5
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
        
        # Extract contexts from response
        contexts = result.get("contexts", {}).get("contexts", [])
        
        return contexts

    def query(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = 0.5
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
