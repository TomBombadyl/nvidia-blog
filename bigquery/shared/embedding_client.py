"""
Embedding Client
Generates embeddings using text-multilingual-embedding-002 model.
"""

import logging
import os
from typing import List
import vertexai
from vertexai.language_models import TextEmbeddingModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating text embeddings."""
    
    def __init__(
        self,
        model_name: str = "text-multilingual-embedding-002",
        project_id: str = None,
        region: str = None,
        embedding_region: str = None
    ):
        """
        Initialize embedding client.
        
        Args:
            model_name: Name of the embedding model (default: text-multilingual-embedding-002)
            project_id: GCP project ID (default: from env or auto-detect)
            region: GCP region for other operations (default: europe-west3)
            embedding_region: Region for embedding model (default: us-central1, where model is available)
        """
        self.model_name = model_name
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "nvidia-blog")
        self.region = region or os.getenv("GCP_REGION", "europe-west3")
        # Embedding models are typically available in us-central1
        self.embedding_region = embedding_region or os.getenv("EMBEDDING_REGION", "us-central1")
        
        # Initialize Vertex AI with correct project and region for embeddings
        # Note: Some Google models are only available in us-central1
        logger.info(
            f"Initializing Vertex AI for embeddings: "
            f"project={self.project_id}, region={self.embedding_region}"
        )
        vertexai.init(project=self.project_id, location=self.embedding_region)
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        logger.info(f"Embedding model loaded: {model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats (768 dimensions)
        """
        try:
            embeddings = self.model.get_embeddings([text])
            
            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from model")
            
            embedding_vector = embeddings[0].values
            
            if len(embedding_vector) != 768:
                logger.warning(
                    f"Expected 768 dimensions, got {len(embedding_vector)}"
                )
            
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (each 768 dimensions)
        """
        try:
            if not texts:
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.get_embeddings(texts)
            
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )
            
            embedding_vectors = [emb.values for emb in embeddings]
            
            logger.info(
                f"Generated {len(embedding_vectors)} embeddings "
                f"({len(embedding_vectors[0]) if embedding_vectors else 0} dimensions each)"
            )
            
            return embedding_vectors
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

