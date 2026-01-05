"""
Read-only Vector Search Query Module
Provides query capabilities for Vertex AI Vector Search.
"""

import logging
from typing import List, Dict
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from vertexai.language_models import TextEmbeddingModel

logger = logging.getLogger(__name__)


class VectorSearchQuery:
    """Read-only query interface for Vertex AI Vector Search."""

    def __init__(
        self,
        endpoint_id: str,
        index_id: str,
        region: str,
        model: str = "text-multilingual-embedding-002"
    ):
        """
        Initialize Vector Search query interface.
        
        Args:
            endpoint_id: Vector Search endpoint ID
            index_id: Vector Search index ID
            region: GCP region
            model: Embedding model name (default: text-multilingual-embedding-002)
        """
        self.endpoint_id = endpoint_id
        self.index_id = index_id
        self.region = region
        self.model = model
        
        aiplatform.init(location=region)
        
        # Get project ID
        project = aiplatform.initializer.global_config.project
        
        # Initialize index
        index_name = (
            f"projects/{project}/locations/{region}/indexes/{index_id}"
        )
        self.index = matching_engine.MatchingEngineIndex(
            index_name=index_name
        )

        # Initialize embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(model)

        # Get deployed index ID from endpoint
        # Initialize endpoint to get deployed index info
        endpoint_name = (
            f"projects/{project}/locations/{region}/"
            f"indexEndpoints/{endpoint_id}"
        )
        self.endpoint = matching_engine.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_name
        )

        # Get deployed index ID
        deployed_indexes = self.endpoint.deployed_indexes
        if not deployed_indexes:
            raise ValueError(
                f"No deployed indexes found on endpoint {endpoint_id}"
            )

        self.deployed_index_id = deployed_indexes[0].id

        logger.info(
            f"Initialized Vector Search query interface - "
            f"Endpoint: {endpoint_id}, Index: {index_id}, "
            f"Deployed Index: {self.deployed_index_id}"
        )
    
    def _embed_query(self, query_text: str) -> List[float]:
        """
        Generate embedding for query text.
        
        Args:
            query_text: Text to embed
            
        Returns:
            Embedding vector as list of floats (768 dimensions)
        """
        try:
            embeddings = self.embedding_model.get_embeddings([query_text])
            
            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from model")
            
            return embeddings[0].values
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        num_neighbors: int = 10
    ) -> Dict:
        """
        Query Vector Search index for similar vectors.
        
        Args:
            query_text: The query text to search for
            num_neighbors: Number of nearest neighbors to return (default: 10)
            
        Returns:
            Dictionary containing query results with neighbors
        """
        try:
            logger.info(f"Querying Vector Search: '{query_text[:50]}...'")
            
            # Generate query embedding
            query_embedding = self._embed_query(query_text)
            
            # Query the index using the endpoint's find_neighbors method
            # The endpoint object has the find_neighbors method, not the index
            results = self.endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors
            )

            # Process results
            neighbors = []
            if results and len(results) > 0:
                neighbor_list = results[0]  # First query result
                for neighbor in neighbor_list:
                    # Preview first 10 dimensions of feature vector
                    vec_preview = (
                        neighbor.feature_vector[:10]
                        if neighbor.feature_vector else []
                    )
                    neighbors.append({
                        "datapoint_id": neighbor.id,
                        "distance": neighbor.distance,
                        "feature_vector": vec_preview
                    })
            
            logger.info(f"Retrieved {len(neighbors)} neighbors from Vector Search")
            
            return {
                "query": query_text,
                "neighbors": neighbors,
                "count": len(neighbors)
            }
            
        except Exception as e:
            logger.error(f"Error querying Vector Search: {e}")
            raise
