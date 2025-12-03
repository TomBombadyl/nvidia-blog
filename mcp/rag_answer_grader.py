"""
Answer Grading Module for RAG Pipeline
Evaluates the quality and relevance of retrieved contexts and generated answers.
"""

import logging
from typing import Dict, List, Optional
from vertexai.generative_models import GenerativeModel
import vertexai
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnswerGrade(BaseModel):
    """Structured grade for an answer or retrieved context."""
    score: float = Field(..., description="Quality score from 0.0 to 1.0")
    relevance: float = Field(..., description="Relevance to query from 0.0 to 1.0")
    completeness: float = Field(..., description="Completeness of answer from 0.0 to 1.0")
    grounded: bool = Field(..., description="Whether answer is grounded in retrieved contexts")
    reasoning: str = Field(..., description="Brief explanation of the grade")
    should_refine: bool = Field(..., description="Whether query should be refined and retried")


class AnswerGrader:
    """Grades retrieved contexts and answers for quality and relevance."""
    
    def __init__(self, project_id: str, region: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize answer grader.
        
        Args:
            project_id: GCP project ID
            region: GCP region
            model_name: Vertex AI model name for grading
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        
        # Initialize generative model
        self.model = GenerativeModel(model_name)
        
        # Quality thresholds
        self.min_score_threshold = 0.6  # Minimum overall score to accept
        self.min_relevance_threshold = 0.65  # Minimum relevance to query
        self.min_completeness_threshold = 0.55  # Minimum completeness
        
        logger.info(f"Initialized AnswerGrader with model: {model_name}")
    
    def grade_contexts(
        self,
        query: str,
        contexts: List[Dict],
        min_acceptable_score: float = 0.6
    ) -> AnswerGrade:
        """
        Grade retrieved contexts for quality and relevance to the query.
        
        Args:
            query: The original user query
            contexts: List of retrieved context dictionaries with 'text' and optionally 'source_uri'
            min_acceptable_score: Minimum score threshold (default: 0.6)
        
        Returns:
            AnswerGrade with scores and refinement recommendation
        """
        try:
            if not contexts:
                return AnswerGrade(
                    score=0.0,
                    relevance=0.0,
                    completeness=0.0,
                    grounded=False,
                    reasoning="No contexts retrieved",
                    should_refine=True
                )
            
            # Prepare context summary for grading
            context_texts = []
            for i, ctx in enumerate(contexts[:5], 1):  # Grade top 5 contexts
                text = ctx.get("text", ctx.get("content", ""))[:500]  # First 500 chars
                source = ctx.get("source_uri", ctx.get("uri", "unknown"))
                context_texts.append(f"[Context {i} from {source}]:\n{text}\n")
            
            contexts_summary = "\n".join(context_texts)
            
            grading_prompt = f"""You are a quality assessment expert for RAG (Retrieval-Augmented Generation) systems specializing in NVIDIA technical documentation.

Your role is to evaluate whether retrieved contexts are sufficient to answer a user's query with high quality, grounded information.

EVALUATION CRITERIA:
1. **Relevance** (0.0-1.0): How well do the contexts address the user's query?
   - 0.9-1.0: Directly answers the query with specific information
   - 0.7-0.89: Relevant but may need minor refinement
   - 0.5-0.69: Partially relevant, some gaps
   - 0.0-0.49: Not relevant or off-topic

2. **Completeness** (0.0-1.0): Do the contexts provide enough information?
   - 0.9-1.0: Complete answer with all necessary details
   - 0.7-0.89: Mostly complete, minor details missing
   - 0.5-0.69: Partial information, significant gaps
   - 0.0-0.49: Insufficient information

3. **Grounded** (true/false): Can the answer be fully supported by the retrieved contexts?
   - true: All claims can be verified from contexts
   - false: Requires information not in contexts (would cause hallucination)

4. **Should Refine** (true/false): Should the query be refined and retried?
   - true: If score < {min_acceptable_score} OR relevance < 0.65 OR completeness < 0.55 OR not grounded
   - false: If contexts are sufficient for a high-quality answer

USER QUERY:
{query}

RETRIEVED CONTEXTS:
{contexts_summary}

Provide your evaluation in this exact JSON format (no markdown, no code blocks):
{{
    "score": <overall_score_0.0_to_1.0>,
    "relevance": <relevance_score_0.0_to_1.0>,
    "completeness": <completeness_score_0.0_to_1.0>,
    "grounded": <true_or_false>,
    "reasoning": "<brief_explanation>",
    "should_refine": <true_or_false>
}}"""

            response = self.model.generate_content(
                grading_prompt,
                generation_config={
                    "temperature": 0.2,  # Very low temperature for consistent grading
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 300,
                }
            )
            
            # Parse JSON response
            import json
            import re
            
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                grade_dict = json.loads(json_match.group())
            else:
                # Fallback: try to parse entire response
                grade_dict = json.loads(response_text)
            
            grade = AnswerGrade(**grade_dict)
            
            logger.info(
                f"Graded contexts for query '{query[:50]}...': "
                f"score={grade.score:.2f}, relevance={grade.relevance:.2f}, "
                f"completeness={grade.completeness:.2f}, grounded={grade.grounded}, "
                f"should_refine={grade.should_refine}"
            )
            
            return grade
            
        except Exception as e:
            logger.error(f"Error grading contexts: {e}")
            # Return a conservative grade that triggers refinement
            return AnswerGrade(
                score=0.5,
                relevance=0.5,
                completeness=0.5,
                grounded=False,
                reasoning=f"Grading error: {str(e)}",
                should_refine=True
            )
