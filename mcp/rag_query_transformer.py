"""
Query Transformation Module for RAG Pipeline
Rewrites weak user questions to improve retrieval quality.
"""

import logging
from typing import Optional
from vertexai.generative_models import GenerativeModel
import vertexai

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Transforms user queries to improve retrieval quality."""
    
    def __init__(self, project_id: str, region: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize query transformer.
        
        Args:
            project_id: GCP project ID
            region: GCP region
            model_name: Vertex AI model name for query transformation
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        
        # Initialize generative model
        self.model = GenerativeModel(model_name)
        
        logger.info(f"Initialized QueryTransformer with model: {model_name}")
    
    def transform_query(
        self,
        original_query: str,
        max_iterations: int = 1
    ) -> str:
        """
        Transform a user query to improve retrieval quality.
        
        This method rewrites weak, vague, or ambiguous queries into more
        specific, searchable queries that will retrieve better results from
        the RAG corpus.
        
        Args:
            original_query: The original user query
            max_iterations: Maximum number of transformation iterations (default: 1)
        
        Returns:
            Transformed query string optimized for retrieval
        """
        try:
            transformation_prompt = f"""You are a query transformation expert specializing in NVIDIA developer documentation and technical content.

Your role is to rewrite user queries to maximize retrieval quality from a RAG (Retrieval-Augmented Generation) system containing NVIDIA blog posts, tutorials, and technical documentation.

TRANSFORMATION GUIDELINES:
1. Expand abbreviations and acronyms (e.g., "CUDA" → "CUDA parallel computing platform")
2. Add technical context when queries are vague (e.g., "optimization" → "CUDA kernel optimization techniques")
3. Include relevant NVIDIA technologies, frameworks, or tools when implied
4. Preserve the user's intent while making the query more specific
5. Use technical terminology that matches NVIDIA documentation style
6. If the query is already well-formed, return it with minimal changes

ORIGINAL QUERY:
{original_query}

TRANSFORMED QUERY (output only the transformed query, no explanations):"""

            response = self.model.generate_content(
                transformation_prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent transformations
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 200,
                }
            )
            
            transformed_query = response.text.strip()
            
            # Fallback to original if transformation fails or is empty
            if not transformed_query or len(transformed_query) < 3:
                logger.warning(
                    f"Query transformation returned empty result, using original query"
                )
                return original_query
            
            logger.info(
                f"Query transformed: '{original_query[:50]}...' → '{transformed_query[:50]}...'"
            )
            
            return transformed_query
            
        except Exception as e:
            logger.error(f"Error transforming query: {e}")
            # Fallback to original query on error
            return original_query
