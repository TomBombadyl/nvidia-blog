"""
Large Content Handler Module

Provides utilities for handling blog posts that exceed embedding API token limits.
Implements dynamic batch sizing, adaptive chunking, and progressive processing.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy based on content size."""
    chunk_size_tokens: int
    overlap_tokens: int
    max_tokens_per_batch: int = 20000
    max_tokens_per_chunk: int = 2048
    max_chunks_per_batch: int = 250


class LargeContentHandler:
    """Handles large content that may exceed embedding API limits."""
    
    # Size thresholds (in bytes)
    LARGE_POST_THRESHOLD = 200 * 1024  # 200KB
    VERY_LARGE_POST_THRESHOLD = 400 * 1024  # 400KB
    
    def __init__(self):
        """Initialize large content handler."""
        self.normal_config = ChunkingConfig(
            chunk_size_tokens=1000,
            overlap_tokens=200
        )
        self.large_config = ChunkingConfig(
            chunk_size_tokens=750,
            overlap_tokens=150
        )
        self.very_large_config = ChunkingConfig(
            chunk_size_tokens=500,
            overlap_tokens=100
        )
    
    def determine_chunking_config(self, content_size_bytes: int) -> ChunkingConfig:
        """
        Determine optimal chunking configuration based on content size.
        
        Args:
            content_size_bytes: Size of content in bytes
        
        Returns:
            ChunkingConfig with appropriate chunk size and overlap
        """
        if content_size_bytes >= self.VERY_LARGE_POST_THRESHOLD:
            logger.info(
                f"Content size {content_size_bytes / 1024:.1f}KB exceeds "
                f"{self.VERY_LARGE_POST_THRESHOLD / 1024:.1f}KB threshold. "
                f"Using very large post configuration."
            )
            return self.very_large_config
        elif content_size_bytes >= self.LARGE_POST_THRESHOLD:
            logger.info(
                f"Content size {content_size_bytes / 1024:.1f}KB exceeds "
                f"{self.LARGE_POST_THRESHOLD / 1024:.1f}KB threshold. "
                f"Using large post configuration."
            )
            return self.large_config
        else:
            return self.normal_config
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Uses estimation: ~4 characters per token for English text.
        For production, should use Vertex AI CountTokens API.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Estimated token count
        """
        # Rough estimation: 4 characters per token
        # This is a conservative estimate for English text
        # For production, integrate with Vertex AI CountTokens API
        return len(text) // 4
    
    def create_embedding_batches(
        self,
        chunks: List[Dict],
        config: Optional[ChunkingConfig] = None
    ) -> List[List[Dict]]:
        """
        Group chunks into batches that respect token limits.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            config: ChunkingConfig with limits (uses default if None)
        
        Returns:
            List of batches, each containing chunks that fit within token limit
        """
        if config is None:
            config = self.normal_config
        
        batches = []
        current_batch = []
        current_token_count = 0
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = self.count_tokens(chunk_text)
            
            # Safety check: ensure no single chunk exceeds per-chunk limit
            if chunk_tokens > config.max_tokens_per_chunk:
                chunk_index = chunk.get('chunk_index', 'unknown')
                logger.warning(
                    f"Chunk {chunk_index} exceeds {config.max_tokens_per_chunk} tokens "
                    f"({chunk_tokens}). This should not happen with proper chunking. "
                    f"Consider reducing chunk size."
                )
                # Truncate to safe limit (this is a fallback, should be rare)
                chunk['text'] = self._truncate_to_tokens(
                    chunk_text,
                    config.max_tokens_per_chunk
                )
                chunk_tokens = config.max_tokens_per_chunk
            
            # Check if adding this chunk would exceed batch limit
            if (current_token_count + chunk_tokens > config.max_tokens_per_batch or
                len(current_batch) >= config.max_chunks_per_batch):
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                    logger.debug(
                        f"Created batch with {len(current_batch)} chunks "
                        f"({current_token_count} tokens)"
                    )
                current_batch = [chunk]
                current_token_count = chunk_tokens
            else:
                # Add to current batch
                current_batch.append(chunk)
                current_token_count += chunk_tokens
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
            logger.debug(
                f"Created final batch with {len(current_batch)} chunks "
                f"({current_token_count} tokens)"
            )
        
        logger.info(
            f"Created {len(batches)} batches from {len(chunks)} chunks "
            f"(avg {len(chunks)/len(batches):.1f} chunks per batch)"
        )
        
        return batches
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to approximately max_tokens.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
        
        Returns:
            Truncated text
        """
        # Estimate: 4 characters per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        
        # Truncate at word boundary if possible
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.9:  # If space is near end
            return truncated[:last_space] + '...'
        return truncated + '...'
    
    def validate_chunks(self, chunks: List[Dict], config: ChunkingConfig) -> Dict[str, any]:
        """
        Validate chunks before processing.
        
        Args:
            chunks: List of chunks to validate
            config: ChunkingConfig with limits
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = self.count_tokens(chunk_text)
            total_tokens += chunk_tokens
            
            if chunk_tokens > config.max_tokens_per_chunk:
                issues.append({
                    'chunk_index': chunk.get('chunk_index', 'unknown'),
                    'tokens': chunk_tokens,
                    'limit': config.max_tokens_per_chunk,
                    'issue': 'exceeds_per_chunk_limit'
                })
        
        # Estimate batches needed
        estimated_batches = (total_tokens + config.max_tokens_per_batch - 1) // config.max_tokens_per_batch
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'estimated_batches': estimated_batches,
            'issues': issues,
            'is_valid': len(issues) == 0
        }


# Convenience function for easy import
def create_batches_for_embedding(
    chunks: List[Dict],
    content_size_bytes: Optional[int] = None
) -> List[List[Dict]]:
    """
    Create batches for embedding API that respect token limits.
    
    Args:
        chunks: List of chunk dictionaries
        content_size_bytes: Optional content size for adaptive chunking
    
    Returns:
        List of batches ready for embedding API
    """
    handler = LargeContentHandler()
    
    if content_size_bytes:
        config = handler.determine_chunking_config(content_size_bytes)
    else:
        config = None
    
    return handler.create_embedding_batches(chunks, config)
