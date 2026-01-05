"""
Query Transformation Module for RAG Pipeline
Rewrites weak user questions to improve retrieval quality.
"""

import logging
from vertexai.generative_models import GenerativeModel
import vertexai
from config import GEMINI_MODEL_NAME, GEMINI_MODEL_LOCATION

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Transforms user queries to improve retrieval quality."""
    
    def __init__(
        self,
        project_id: str,
        region: str,
        model_name: str = None,
        gemini_location: str = None
    ):
        """
        Initialize query transformer.

        Args:
            project_id: GCP project ID
            region: GCP region (for RAG corpus, not used for Gemini)
            model_name: Vertex AI model name for query transformation
            gemini_location: Location for Gemini model
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name or GEMINI_MODEL_NAME
        self.gemini_location = gemini_location or GEMINI_MODEL_LOCATION
        
        # Initialize Vertex AI with europe-west4 for Gemini models
        # Using europe-west4 (Netherlands) - closest European region
        # to RAG corpus (europe-west3)
        # This follows Google best practice: use region-specific
        # locations for data residency
        vertexai.init(project=project_id, location=self.gemini_location)

        # Initialize generative model
        self.model = GenerativeModel(self.model_name)

        logger.info(
            "Initialized QueryTransformer with model: %s in location: %s",
            self.model_name,
            self.gemini_location
        )
    
    def transform_query(
        self,
        original_query: str,
        max_iterations: int = 1
    ) -> str:
        """
        Transform a user query to improve retrieval quality.

        This method rewrites weak, vague, or ambiguous queries into
        more specific, searchable queries that will retrieve better
        results from the RAG corpus.

        Args:
            original_query: The original user query
            max_iterations: Maximum number of transformation iterations

        Returns:
            Transformed query string optimized for retrieval
        """
        try:
            # Get current date for temporal query context
            from datetime import datetime
            current_date = datetime.utcnow().strftime("%B %d, %Y")
            current_month_year = datetime.utcnow().strftime("%B %Y")

            # Build transformation prompt with date awareness
            transformation_prompt = (
                "You are a query transformation expert specializing in "
                "NVIDIA developer documentation and technical content.\n\n"
                "Your role is to rewrite user queries to maximize retrieval "
                "quality from a RAG system containing NVIDIA blog posts, "
                "tutorials, and technical documentation.\n\n"
                f"CURRENT DATE CONTEXT:\n"
                f"Today's date is {current_date}. The corpus contains blog "
                f"posts with publication dates embedded in the text as "
                f"'Publication Date: [date]'.\n\n"
                "TRANSFORMATION GUIDELINES:\n"
                "1. Preserve the user's original intent and topic focus\n"
                "2. Expand abbreviations and acronyms only when clearly needed\n"
                "3. Add minimal technical context when queries are genuinely vague\n"
                "4. Include NVIDIA-specific terms only when clearly implied\n"
                "5. Use technical terminology matching NVIDIA docs when appropriate\n"
                "6. If well-formed or specific, return with minimal or no changes\n"
                "7. Avoid adding assumptions about what the user wants\n\n"
                "TEMPORAL QUERY HANDLING:\n"
                "- For 'today', 'recent', 'latest', 'newest': Add month/year context "
                f"({current_month_year}) to help with temporal filtering\n"
                "- Keep the query focused on what the user asked about\n"
                "- Add 'NVIDIA' as context only if not already present\n"
                "- Do NOT inject topic assumptions (GPU, CUDA, AI, etc.) unless "
                "the user's query clearly implies them\n"
                f"- Example: 'What's new today?' → 'What's new from NVIDIA in {current_month_year}'\n"
                f"- Example: 'Recent GPU developments' → 'Recent NVIDIA GPU developments from {current_month_year}'\n"
                f"- Example: 'Latest CUDA tips' → 'Latest CUDA programming tips from {current_month_year}'\n"
                "- Avoid queries that only match metadata headers - if query is "
                "date-only, suggest adding 'blog posts' or 'articles' for better matching\n\n"
                "QUERY TYPE EXAMPLES:\n"
                "- Specific technical: 'CUDA optimization' → 'CUDA optimization techniques' (minimal change)\n"
                "- Vague: 'optimization tips' → 'NVIDIA optimization tips and best practices'\n"
                "- Product-specific: 'Omniverse updates' → 'NVIDIA Omniverse updates' (add brand only)\n"
                f"- Temporal vague: 'latest news' → 'latest NVIDIA news from {current_month_year}'\n"
                "- Already good: 'TensorRT inference performance' → (return as-is or minimal change)\n\n"
                f"ORIGINAL QUERY:\n{original_query}\n\n"
                "TRANSFORMED QUERY (output only the transformed query, "
                "no explanations):"
            )

            response = self.model.generate_content(
                transformation_prompt,
                generation_config={
                    "temperature": 0.4,  # Balanced temperature for flexible yet consistent transformations
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 200,
                }
            )
            
            transformed_query = response.text.strip()
            
            # Fallback to original if transformation fails or is empty
            if not transformed_query or len(transformed_query) < 3:
                logger.warning(
                    "Query transformation returned empty result, "
                    "using original query"
                )
                return original_query

            logger.info(
                "Query transformed: '%s...' → '%s...'",
                original_query[:50],
                transformed_query[:50]
            )

            return transformed_query

        except Exception as e:
            logger.error("Error transforming query: %s", e)
            # Fallback to original query on error
            return original_query
