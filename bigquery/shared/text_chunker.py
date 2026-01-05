"""
Text Chunking Utility
Replicates RAG Corpus chunking logic (768 words, 128 overlap).
"""

import re
from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 768,
    overlap: int = 128
) -> List[Dict]:
    """
    Chunk text into word-based chunks with overlap.
    Matches RAG Corpus chunking exactly.
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk (default: 768)
        overlap: Number of words to overlap between chunks (default: 128)
    
    Returns:
        List of dicts with 'text', 'chunk_index', 'start_word', 'end_word', 'word_count'
    """
    # Split text into words (preserve whitespace for reconstruction)
    words = text.split()
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'chunk_index': len(chunks),
            'start_word': start_idx,
            'end_word': end_idx,
            'word_count': len(chunk_words)
        })
        
        # Move start forward by (chunk_size - overlap)
        start_idx += (chunk_size - overlap)
        
        # Break if we've reached the end
        if end_idx >= len(words):
            break
    
    return chunks

