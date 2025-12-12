"""
Multilingual Response Generator for RAG Pipeline
Generates responses in the query language based on retrieved contexts.
"""

import logging
from typing import Dict, List, Optional
from vertexai.generative_models import GenerativeModel
import vertexai
from config import GEMINI_MODEL_NAME, GEMINI_MODEL_LOCATION

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates multilingual responses from retrieved contexts."""
    
    def __init__(self, project_id: str, region: str, model_name: str = None, gemini_location: str = None):
        """
        Initialize response generator.
        
        Args:
            project_id: GCP project ID
            region: GCP region (for RAG corpus, not used for Gemini)
            model_name: Vertex AI model name (default: from config)
            gemini_location: Location for Gemini model (default: from config)
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name or GEMINI_MODEL_NAME
        self.gemini_location = gemini_location or GEMINI_MODEL_LOCATION
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=self.gemini_location)
        
        # Initialize generative model
        self.model = GenerativeModel(self.model_name)
        
        logger.info(
            f"Initialized ResponseGenerator with model {self.model_name} "
            f"in {self.gemini_location}"
        )
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the query text.
        Returns ISO 639-1 language code (e.g., 'es', 'fr', 'de', 'ja', 'zh').
        Falls back to 'en' if detection fails.
        """
        try:
            # Simple heuristic-based detection for common languages
            # For production, consider using a proper language detection library
            text_lower = text.lower()
            
            # Spanish indicators
            if any(word in text_lower for word in ['cómo', 'qué', 'cuál', 'para', 'con', 'por', 'español']):
                return 'es'
            
            # French indicators
            if any(word in text_lower for word in ['comment', 'pourquoi', 'quand', 'où', 'français']):
                return 'fr'
            
            # German indicators
            if any(word in text_lower for word in ['wie', 'was', 'wo', 'wann', 'warum', 'deutsch']):
                return 'de'
            
            # Japanese indicators (hiragana/katakana/kanji)
            if any(ord(char) >= 0x3040 and ord(char) <= 0x9FFF for char in text):
                return 'ja'
            
            # Chinese indicators (simplified/traditional)
            if any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in text):
                return 'zh'
            
            # Default to English
            return 'en'
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return 'en'
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from ISO code."""
        lang_names = {
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'en': 'English'
        }
        return lang_names.get(lang_code, 'English')
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict],
        language: Optional[str] = None
    ) -> str:
        """
        Generate a response in the specified language based on retrieved contexts.
        
        Args:
            query: The original user query
            contexts: List of retrieved context dictionaries with 'text' field
            language: Target language code (ISO 639-1). If None, auto-detects from query.
        
        Returns:
            Generated response text in the target language
        """
        if not contexts:
            return "No relevant information found."
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(query)
        
        lang_name = self.get_language_name(language)
        
        # Prepare context text
        context_texts = []
        for i, ctx in enumerate(contexts[:5], 1):  # Use top 5 contexts
            text = ctx.get('text', '') or ctx.get('content', '')
            if text:
                context_texts.append(f"[Context {i}]\n{text[:1000]}")  # Limit each context to 1000 chars
        
        contexts_combined = "\n\n".join(context_texts)
        
        # Create generation prompt
        if language == 'en':
            generation_prompt = f"""You are a helpful assistant that provides accurate information from NVIDIA developer blog content.

User Query: {query}

Retrieved Contexts from NVIDIA Blogs:
{contexts_combined}

Based on the retrieved contexts above, provide a comprehensive answer to the user's query. 
- Use only information from the provided contexts
- Be accurate and cite specific details when possible
- If the contexts don't fully answer the query, say so
- Write in clear, professional English

Answer:"""
        else:
            generation_prompt = f"""You are a helpful assistant that provides accurate information from NVIDIA developer blog content.

User Query (in {lang_name}): {query}

Retrieved Contexts from NVIDIA Blogs (in English):
{contexts_combined}

Based on the retrieved English contexts above, provide a comprehensive answer to the user's query.
- Answer in {lang_name} (the same language as the user's query)
- Use only information from the provided contexts
- Translate and adapt the English content naturally to {lang_name}
- Be accurate and cite specific details when possible
- If the contexts don't fully answer the query, say so in {lang_name}
- Write in clear, professional {lang_name}

Answer (in {lang_name}):"""
        
        try:
            response = self.model.generate_content(
                generation_prompt,
                generation_config={
                    "temperature": 0.3,  # Balanced temperature for natural but accurate responses
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2000,  # Allow longer responses for comprehensive answers
                }
            )
            
            generated_text = response.text.strip()
            
            if not generated_text:
                logger.warning("Generated response is empty")
                return "I couldn't generate a response. Please try rephrasing your query."
            
            logger.info(
                f"Generated response in {lang_name} for query: '{query[:50]}...' "
                f"(length: {len(generated_text)} chars)"
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

