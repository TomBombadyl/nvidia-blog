"""
MCP Server for NVIDIA Blog BigQuery Search
Provides read-only query access to BigQuery for semantic search.
This server does NOT modify the database - only the ingestion jobs can write data.
"""

import logging
import re
import sys
from typing import Optional, Union, List, Any, Dict
from pydantic import BaseModel, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
import os

# Import FastMCP first (from installed package)
from mcp.server.fastmcp import FastMCP

# Add parent directory to import models from existing mcp_server
parent_dir = os.path.join(os.path.dirname(__file__), '..', '..')
parent_dir = os.path.abspath(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import models from existing mcp_server using importlib with proper context
import importlib.util

# First, load config module and make it available for relative imports
config_path = os.path.join(parent_dir, 'archive', 'mcp-rag-corpus', 'config.py')
config_spec = importlib.util.spec_from_file_location("mcp.config", config_path)
config_module = importlib.util.module_from_spec(config_spec)
# Set __package__ so relative imports work
config_module.__package__ = 'mcp'
config_spec.loader.exec_module(config_module)
# Make 'config' available for relative imports in mcp_server
sys.modules['config'] = config_module
sys.modules['mcp.config'] = config_module

# Now load mcp_server module with proper context
mcp_server_path = os.path.join(parent_dir, 'archive', 'mcp-rag-corpus', 'mcp_server.py')
mcp_server_spec = importlib.util.spec_from_file_location("mcp.mcp_server", mcp_server_path)
mcp_server_module = importlib.util.module_from_spec(mcp_server_spec)
# Set __package__ and __file__ for proper module context
mcp_server_module.__package__ = 'mcp'
mcp_server_module.__file__ = mcp_server_path
# Make config available in the module's namespace
mcp_server_module.config = config_module
# Execute the module
mcp_server_spec.loader.exec_module(mcp_server_module)

# Import the models
RAGContext = mcp_server_module.RAGContext
RAGQueryResult = mcp_server_module.RAGQueryResult
ErrorResult = mcp_server_module.ErrorResult

from bigquery.mcp_server.config import (
    PROJECT_ID,
    REGION,
    BIGQUERY_DATASET,
    BIGQUERY_TABLE_CHUNKS,
    RAG_VECTOR_DISTANCE_THRESHOLD
)

# Import query module lazily to prevent startup failures
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server with descriptive name and metadata
mcp = FastMCP(
    "NVIDIA Developer Resources Search",
    stateless_http=True,  # Required for Cloud Run/serverless deployments
    instructions="A read-only search tool for NVIDIA developer blog content. Provides grounded, factual information from official NVIDIA sources with automatic daily updates.",
)

# Add health check endpoint
@mcp.custom_route(path="/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for load balancers and connection verification."""
    return JSONResponse({"status": "healthy", "server": "NVIDIA Developer Resources Search"})

# Initialize query interface (lazy initialization)
_bigquery_query: Optional["BigQueryRAGQuery"] = None


def get_bigquery_query():
    """Get or create BigQuery RAG query interface."""
    global _bigquery_query
    if _bigquery_query is None:
        try:
            # Lazy import to prevent startup failures
            from bigquery.mcp_server.query_bigquery import BigQueryRAGQuery
            logger.info(f"Initializing BigQueryRAGQuery with table: {BIGQUERY_DATASET}.{BIGQUERY_TABLE_CHUNKS}")
            _bigquery_query = BigQueryRAGQuery(
                PROJECT_ID,
                BIGQUERY_DATASET,
                BIGQUERY_TABLE_CHUNKS,
                enable_transformation=True,  # Enable query transformation
                enable_grading=True,  # Enable answer grading and refinement
                max_refinement_iterations=2  # Allow up to 2 refinement iterations
            )
            logger.info("BigQueryRAGQuery initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize BigQueryRAGQuery")
            raise
    return _bigquery_query


def _generate_alternative_queries(original_query: str, transformed_query: Optional[str] = None) -> List[str]:
    """
    Generate alternative query variations to try when initial search returns 0 results.
    
    Args:
        original_query: The original user query
        transformed_query: The transformed query (if available)
    
    Returns:
        List of alternative query variations to try
    """
    alternatives = []
    
    # Use transformed query as base if available, otherwise use original
    base_query = transformed_query if transformed_query else original_query
    
    # Try 1: Add "NVIDIA" if not present
    if "NVIDIA" not in base_query.upper():
        alternatives.append(f"NVIDIA {base_query}")
    
    # Try 2: Simplify to key terms
    words = base_query.split()
    if len(words) > 3:
        # Take first 3-4 important words
        key_terms = [w for w in words[:4] if len(w) > 2 and w.lower() not in ['the', 'a', 'an', 'and', 'or', 'for', 'with', 'from']]
        if key_terms:
            alternatives.append(" ".join(key_terms))
    
    # Try 3: Broaden the query (remove specific terms, keep general ones)
    if base_query != original_query:
        alternatives.append(original_query)  # Try original if we had a transformation
    
    # Try 4: Add "blog" or "article" context
    if "blog" not in base_query.lower() and "article" not in base_query.lower():
        alternatives.append(f"{base_query} blog")
    
    # Try 5: Remove temporal references and try broader search
    temporal_pattern = r'\b(December|January|February|March|April|May|June|July|August|September|October|November)\s+\d{4}\b'
    broad_query = re.sub(temporal_pattern, '', base_query, flags=re.IGNORECASE).strip()
    if broad_query and broad_query != base_query:
        alternatives.append(broad_query)
    
    # Limit to 5 alternatives max
    return alternatives[:5]


@mcp.tool()
def search_nvidia_blogs(
    query: str,
    method: str = "rag",
    top_k: int = 10
) -> dict:
    """
    Enhanced grounded search tool for NVIDIA developer resources with query transformation,
    answer grading, and iterative refinement using BigQuery.

    This is a self-maintained, production-ready search tool that provides grounded,
    factual information from NVIDIA's official blog archives. The content is
    automatically updated daily from both the NVIDIA Developer Blog and Official Blog,
    making it a reliable, up-to-date resource for builders and developers working
    with NVIDIA technologies.

    ENHANCED RAG PIPELINE FEATURES:
    - **Query Transformation**: Automatically rewrites weak or vague queries to improve
      retrieval quality, expanding abbreviations and adding technical context.
    - **Answer Grading**: Evaluates retrieved contexts for relevance, completeness,
      and grounding to ensure high-quality results.
    - **Iterative Refinement**: If initial results are graded poorly, the system
      automatically refines the query and retries retrieval up to 2 times.
    - **Strict Grounding**: All responses are grounded in retrieved contexts from
      official NVIDIA sources, preventing hallucinations.

    The tool uses Retrieval-Augmented Generation (RAG) with BigQuery to search through processed
    blog posts containing tutorials, code examples, best practices, technical guides,
    and announcements. All content is sourced directly from NVIDIA's official blogs,
    ensuring accuracy and authority.

    This tool is designed to be called by other MCP servers or AI agents that need
    grounded, factual information about NVIDIA technologies, eliminating hallucinations
    by providing real, verifiable content from official sources.

    Args:
        query: Your search query describing what you need (e.g., "How to optimize CUDA kernels",
               "TensorRT inference tutorial", "GPU memory management", "Deep learning best practices")
        method: Search method - "rag" (recommended for grounded responses) (default: "rag")
        top_k: Number of results to return, 1-20 (default: 10). More results provide broader context.

    Returns:
        SearchResult containing verified blog content with:
        - Full text chunks from official NVIDIA blog posts
        - Similarity scores indicating relevance

    Use Cases:
        - Finding code examples and tutorials
        - Getting best practices and technical guidance
        - Researching NVIDIA technology features
        - Learning about GPU programming, CUDA, TensorRT, etc.
        - Staying updated with NVIDIA announcements and releases

    Example Queries:
        - "CUDA programming best practices"
        - "How to use TensorRT for inference optimization"
        - "GPU acceleration techniques for deep learning"
        - "NVIDIA GPU memory management"
        - "Multi-GPU training strategies"
        - "TensorFlow with NVIDIA GPUs"

    Note: This tool provides grounded, factual information from official NVIDIA sources.
    Content is automatically maintained and updated daily via scheduled ingestion jobs.
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            return ErrorResult(error="Query text cannot be empty")

        top_k = max(1, min(top_k, 25))  # Clamp between 1 and 25

        if method.lower() == "rag":
            # Use BigQuery RAG method with transformation, grading, and refinement
            bigquery_query = get_bigquery_query()
            result_dict = bigquery_query.query(
                query_text=query,
                similarity_top_k=top_k,
                vector_distance_threshold=RAG_VECTOR_DISTANCE_THRESHOLD
            )

            # Convert dict result to Pydantic model
            contexts = [
                RAGContext.model_validate(ctx) if isinstance(ctx, dict) else ctx
                for ctx in result_dict.get("contexts", [])
            ]

            result = RAGQueryResult(
                query=result_dict.get("query", query),
                transformed_query=result_dict.get("transformed_query"),
                contexts=contexts,
                count=result_dict.get("count", len(contexts)),
                grade=result_dict.get("grade"),
                refinement_iterations=result_dict.get("refinement_iterations", 0)
            )

            # If no results found, try alternative queries
            if result.count == 0:
                logger.info(
                    f"No results found for initial query: '{query[:50]}...'. "
                    f"Trying alternative query variations..."
                )
                
                alternative_queries = _generate_alternative_queries(
                    query,
                    result.transformed_query
                )
                
                # Try up to 5 alternative queries
                for alt_query in alternative_queries[:5]:
                    logger.info(f"Trying alternative query: '{alt_query[:50]}...'")
                    alt_result_dict = bigquery_query.query(
                        query_text=alt_query,
                        similarity_top_k=top_k,
                        vector_distance_threshold=RAG_VECTOR_DISTANCE_THRESHOLD
                    )
                    
                    alt_contexts = [
                        RAGContext.model_validate(ctx) if isinstance(ctx, dict) else ctx
                        for ctx in alt_result_dict.get("contexts", [])
                    ]
                    
                    if len(alt_contexts) > 0:
                        # Found results with alternative query!
                        logger.info(
                            f"Found {len(alt_contexts)} contexts with alternative query: '{alt_query[:50]}...'"
                        )
                        result = RAGQueryResult(
                            query=result.query,  # Keep original query
                            transformed_query=alt_query,  # Show which alternative worked
                            contexts=alt_contexts,
                            count=len(alt_contexts),
                            grade=alt_result_dict.get("grade"),
                            refinement_iterations=alt_result_dict.get("refinement_iterations", 0)
                        )
                        return result
                
                # All alternative queries also returned 0 results
                logger.info(
                    f"All {len(alternative_queries)} alternative queries returned 0 results. "
                    f"Informing user that no content was found."
                )
                result.message = (
                    "I wasn't able to find any relevant content in the indexed NVIDIA blog posts "
                    "for your query. This could mean the topic hasn't been covered yet, or it may "
                    "be in older posts not yet indexed. You can browse the full database of NVIDIA "
                    "blog posts at https://blogs.nvidia.com/ to find what you're looking for."
                )

            logger.info(
                f"BigQuery RAG search complete: '{query[:50]}...' "
                f"returned {result.count} contexts"
            )
            return result

        else:
            return ErrorResult(
                error=f"Invalid method '{method}'. Use 'rag' (BigQuery backend)"
            )

    except Exception as e:
        error_msg = f"Error searching NVIDIA blogs: {str(e)}"
        logger.exception("Error in search_nvidia_blogs tool")  # Log full stack trace
        import traceback
        full_traceback = traceback.format_exc()
        logger.debug(f"Full traceback: {full_traceback}")
        return ErrorResult(error=error_msg)

