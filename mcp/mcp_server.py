"""
MCP Server for NVIDIA Blog RSS Ingestion System
Provides read-only query access to RAG Corpus and Vector Search.
This server does NOT modify the database - only the scheduled ingestion jobs can write data.
"""

import logging
import sys
from typing import Optional, Union, List, Any, Dict
from pydantic import BaseModel, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from config import (
    RAG_CORPUS,
    REGION,
    VECTOR_SEARCH_ENDPOINT_ID,
    VECTOR_SEARCH_INDEX_ID,
    EMBEDDING_MODEL,
    RAG_VECTOR_DISTANCE_THRESHOLD
)
# Import query modules lazily to prevent startup failures
# from query_rag import RAGQuery
# from query_vector_search import VectorSearchQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for structured output validation
class RAGContext(BaseModel):
    """A single context from RAG Corpus query with grounding information."""
    source_uri: Optional[str] = Field(
        default=None, 
        description="Source URI of the context (citation)"
    )
    text: str = Field(
        default="", 
        description="Text content of the context"
    )
    distance: Optional[float] = Field(
        default=None, 
        description="Similarity distance score (lower is better)"
    )
    
    @model_validator(mode='before')
    @classmethod
    def normalize_field_names(cls, data: Any) -> Dict[str, Any]:
        """
        Normalize field names from API response variations.
        Handles both 'text'/'content' and 'source_uri'/'uri' field name variations.
        This is the modern Pydantic V2 approach for handling API response variations.
        """
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        
        # Normalize text field: handle both 'text' and 'content'
        normalized['text'] = data.get('text') or data.get('content') or ""
        
        # Normalize source_uri field: handle both 'source_uri' and 'uri'
        normalized['source_uri'] = (
            data.get('source_uri') or 
            data.get('uri') or 
            None
        )
        
        # Preserve distance if present
        if 'distance' in data:
            normalized['distance'] = data['distance']
        
        return normalized


class RAGQueryResult(BaseModel):
    """Enhanced result from RAG Corpus query with transformation and grading metadata."""
    query: str = Field(..., description="The original user query text")
    transformed_query: Optional[str] = Field(None, description="Transformed query used for retrieval (if transformation was applied)")
    contexts: List[RAGContext] = Field(
        default_factory=list,
        description="List of relevant text chunks with metadata from official NVIDIA blog content"
    )
    count: int = Field(..., description="Number of contexts returned")
    grade: Optional[dict] = Field(
        None,
        description="Quality grade with score, relevance, completeness, grounded status, and reasoning (if grading was enabled)"
    )
    refinement_iterations: int = Field(
        0,
        description="Number of query refinement iterations performed (0 = no refinement needed)"
    )


class VectorNeighbor(BaseModel):
    """A single neighbor from Vector Search query."""
    datapoint_id: str = Field(..., description="Unique identifier of the datapoint")
    distance: float = Field(..., description="Distance score (lower is more similar)")
    feature_vector: List[float] = Field(
        default_factory=list,
        description="Preview of feature vector (first 10 dimensions)"
    )


class VectorSearchResult(BaseModel):
    """Result from Vector Search query."""
    query: str = Field(..., description="The original query text")
    neighbors: List[VectorNeighbor] = Field(
        default_factory=list,
        description="List of similar documents with distance scores"
    )
    count: int = Field(..., description="Number of neighbors returned")


class ErrorResult(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message describing what went wrong")


# Union type for search results that can return either RAG or Vector results
SearchResult = Union[RAGQueryResult, VectorSearchResult, ErrorResult]

# Initialize MCP server with descriptive name and metadata
# stateless_http=True is required for Cloud Run deployment
# FastMCP automatically generates serverInfo with name and SDK version in initialize response
mcp = FastMCP(
    "NVIDIA Developer Resources Search",
    stateless_http=True,  # Required for Cloud Run/serverless deployments
    instructions="A read-only search tool for NVIDIA developer blog content. Provides grounded, factual information from official NVIDIA sources with automatic daily updates.",
    # Optional: Add website URL if you have one
    # website_url="https://developer.nvidia.com"
)

# Use default streamable_http_path ("/mcp")
# When mounted at "/" in Starlette, the endpoint will be at "/mcp" (mount path + default path)
# mcp.settings.streamable_http_path = "/mcp"  # This is the default

# Add health check endpoint for faster connection verification
# This helps with cold starts and connection testing
@mcp.custom_route(path="/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for load balancers and connection verification."""
    return JSONResponse({"status": "healthy", "server": "NVIDIA Developer Resources Search"})

# Initialize query interfaces (lazy initialization)
# Use string annotations since these classes are imported lazily
_rag_query: Optional["RAGQuery"] = None
_vector_query: Optional["VectorSearchQuery"] = None


def get_rag_query():
    """Get or create enhanced RAG query interface with transformation and grading."""
    global _rag_query
    if _rag_query is None:
        try:
            # Lazy import to prevent startup failures
            from query_rag import RAGQuery
            logger.info(f"Initializing RAGQuery with corpus: {RAG_CORPUS}")
            _rag_query = RAGQuery(
                RAG_CORPUS,
                REGION,
                enable_transformation=True,  # Enable query transformation
                enable_grading=True,  # Enable answer grading and refinement
                max_refinement_iterations=2  # Allow up to 2 refinement iterations
            )
            logger.info("RAGQuery initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize RAGQuery")
            raise
    return _rag_query


def get_vector_query():
    """Get or create Vector Search query interface."""
    global _vector_query
    if _vector_query is None:
        try:
            # Lazy import to prevent startup failures
            from query_vector_search import VectorSearchQuery
            logger.info(f"Initializing VectorSearchQuery with endpoint: {VECTOR_SEARCH_ENDPOINT_ID}")
            _vector_query = VectorSearchQuery(
                VECTOR_SEARCH_ENDPOINT_ID,
                VECTOR_SEARCH_INDEX_ID,
                REGION,
                EMBEDDING_MODEL
            )
            logger.info("VectorSearchQuery initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize VectorSearchQuery")
            raise
    return _vector_query


@mcp.tool()
def search_nvidia_blogs(
    query: str,
    method: str = "rag",
    top_k: int = 10
) -> dict:
    # Returns dict (Pydantic models serialize to dict)
    # Actual return types: RAGQueryResult | VectorSearchResult | ErrorResult
    """
    Enhanced grounded search tool for NVIDIA developer resources with query transformation,
    answer grading, and iterative refinement.

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

    The tool uses Retrieval-Augmented Generation (RAG) to search through processed
    blog posts containing tutorials, code examples, best practices, technical guides,
    and announcements. All content is sourced directly from NVIDIA's official blogs,
    ensuring accuracy and authority.

    This tool is designed to be called by other MCP servers or AI agents that need
    grounded, factual information about NVIDIA technologies, eliminating hallucinations
    by providing real, verifiable content from official sources.

    Search Methods:
    - "rag" (default): Retrieval-Augmented Generation - Returns full text chunks with
      context from blog posts. Best for most queries as it provides complete information.
    - "vector": Semantic similarity search - Finds conceptually similar content using
      embeddings. Useful when you need to find related concepts even without exact keywords.

    Args:
        query: Your search query describing what you need (e.g., "How to optimize CUDA kernels",
               "TensorRT inference tutorial", "GPU memory management", "Deep learning best practices")
        method: Search method - "rag" (recommended for grounded responses) or "vector" (default: "rag")
        top_k: Number of results to return, 1-20 (default: 10). More results provide broader context.

    Returns:
        SearchResult containing verified blog content with:
        - Full text chunks from official NVIDIA blog posts (RAG method)
        - Or semantic similarity matches with document IDs (Vector method)
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

        top_k = max(1, min(top_k, 20))  # Clamp between 1 and 20

        if method.lower() == "rag":
            # Use enhanced RAG method with transformation, grading, and refinement
            rag_query = get_rag_query()
            result_dict = rag_query.query(
                query_text=query,
                similarity_top_k=top_k,
                vector_distance_threshold=RAG_VECTOR_DISTANCE_THRESHOLD
            )

            # Convert dict result to Pydantic model
            # The RAGContext model_validator handles field normalization automatically
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

            logger.info(
                f"RAG search successful: '{query[:50]}...' "
                f"returned {result.count} contexts"
            )
            return result

        elif method.lower() == "vector":
            # Use Vector Search method - returns semantic similarity matches
            vector_query = get_vector_query()
            result_dict = vector_query.query(
                query_text=query,
                num_neighbors=top_k
            )

            # Convert dict result to Pydantic model
            neighbors = [
                VectorNeighbor(**nb) if isinstance(nb, dict) else nb
                for nb in result_dict.get("neighbors", [])
            ]

            result = VectorSearchResult(
                query=result_dict.get("query", query),
                neighbors=neighbors,
                count=result_dict.get("count", len(neighbors))
            )

            logger.info(
                f"Vector search successful: '{query[:50]}...' "
                f"returned {result.count} neighbors"
            )
            return result

        else:
            return ErrorResult(
                error=f"Invalid method '{method}'. Use 'rag' or 'vector'"
            )

    except Exception as e:
        error_msg = f"Error searching NVIDIA blogs: {str(e)}"
        logger.exception("Error in search_nvidia_blogs tool")  # Log full stack trace
        import traceback
        full_traceback = traceback.format_exc()
        logger.debug(f"Full traceback: {full_traceback}")
        return ErrorResult(error=error_msg)


# NOTE: streamable_http_app() is created in mcp_service.py inside the lifespan
# This ensures the session manager's task group is initialized before the app handles requests
# This server is only deployed to Cloud Run, not run locally
