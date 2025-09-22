"""
Map-Reduce Extractor for processing large corpora using batch-based LLM analysis
Implements the map-reduce strategy to analyze 100% of content while maintaining efficiency
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
from datetime import datetime

from langchain.schema import HumanMessage
from langchain_litellm import ChatLiteLLM
from tqdm import tqdm

from ..data.models.persona_constitution import MentalModel, CoreBelief
from ..config.settings import Settings
from ..core.extractor_cache import ExtractorCacheManager
from ..utils.logging import get_logger


class MapReduceExtractor:
    """
    Map-Reduce extractor for processing large document corpora using batch-based analysis
    
    Maps: Process document batches to extract candidate mental models and core beliefs
    Reduce: Consolidate and synthesize candidates into final, high-quality results
    """
    
    def __init__(self, settings: Settings, persona_id: Optional[str] = None):
        """Initialize the map-reduce extractor"""
        self.settings = settings
        self.persona_id = persona_id
        self.logger = get_logger(__name__)
        
        # Initialize cache manager
        self.cache_manager = ExtractorCacheManager(settings, persona_id)
        
        # Initialize LLMs for different phases
        self.map_llm = None
        self.reduce_llm = None
        self._init_llms()
        
        # Initialize prompts
        self.prompts = self._init_prompts()
        
        # Processing stats
        self.processing_stats = {
            "total_batches": 0,
            "completed_batches": 0,
            "cached_batches": 0,
            "failed_batches": 0,
            "total_processing_time": 0.0
        }
    
    def _init_llms(self):
        """Initialize LLMs for map and reduce phases"""
        import os
        
        config = self.settings.map_reduce_extraction
        
        # Determine API key based on model provider
        if config.map_phase_model.startswith('openrouter/'):
            # Using OpenRouter
            llm_config = self.settings.get_llm_config()
            api_key = llm_config.get('api_key')
            api_key_param = "openrouter_api_key"
        elif config.map_phase_model.startswith('gemini/'):
            # Using Gemini directly
            api_key = os.getenv('GEMINI_API_KEY')
            api_key_param = "api_key"
        else:
            # Other providers
            api_key = None
            api_key_param = None
        
        # Map phase LLM (for batch processing)
        try:
            llm_kwargs = {
                "model": config.map_phase_model,
                "temperature": 0.7,  # Slightly lower for more consistent batch results
                "max_tokens": 4000,
                "timeout": config.timeout_seconds,
                "max_retries": config.max_retries
            }
            
            # Add API key if available
            if api_key and api_key_param:
                llm_kwargs[api_key_param] = api_key
            
            self.map_llm = ChatLiteLLM(**llm_kwargs)
            self.logger.info(f"Initialized map phase LLM: {config.map_phase_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize map phase LLM: {e}")
            raise
        
        # Reduce phase LLM (for consolidation)
        try:
            # Use same API key logic for reduce phase
            if config.reduce_phase_model.startswith('openrouter/'):
                llm_config = self.settings.get_llm_config()
                reduce_api_key = llm_config.get('api_key')
                reduce_api_key_param = "openrouter_api_key"
            elif config.reduce_phase_model.startswith('gemini/'):
                reduce_api_key = os.getenv('GEMINI_API_KEY')
                reduce_api_key_param = "api_key"
            else:
                reduce_api_key = None
                reduce_api_key_param = None
            
            llm_kwargs = {
                "model": config.reduce_phase_model,
                "temperature": 0.3,  # Lower temperature for more deterministic consolidation
                "max_tokens": 6000,  # Higher for consolidation tasks
                "timeout": config.timeout_seconds * 2,  # Longer timeout for complex consolidation
                "max_retries": config.max_retries
            }
            
            # Add API key if available
            if reduce_api_key and reduce_api_key_param:
                llm_kwargs[reduce_api_key_param] = reduce_api_key
            
            self.reduce_llm = ChatLiteLLM(**llm_kwargs)
            self.logger.info(f"Initialized reduce phase LLM: {config.reduce_phase_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize reduce phase LLM: {e}")
            raise
    
    def _init_prompts(self) -> Dict[str, str]:
        """Initialize extraction prompts for map and reduce phases"""
        
        # Map phase prompt for mental models
        map_mental_models_prompt = """
You are analyzing a batch of content from an influencer to identify MENTAL MODELS and FRAMEWORKS.

CONTENT BATCH:
{content}

STATISTICAL INSIGHTS:
{statistical_insights}

Your task is to extract CANDIDATE mental models from this batch. Focus on:
1. Systematic approaches to problems mentioned in this content
2. Step-by-step methodologies described
3. Frameworks with clear stages
4. Problem-solving patterns shown
5. Decision-making processes outlined

For each mental model found, provide:
- NAME: Clear, descriptive name
- DESCRIPTION: What the model does
- STEPS: Numbered steps (2-10 steps)
- APPLICATION_CONTEXTS: Where it's used
- EXAMPLES: Real examples from this batch
- FREQUENCY_SCORE: How often it appears in this batch (1-10)

Return a JSON array of mental models:
[
    {{
        "name": "Framework Name",
        "description": "Brief description of what this framework does",
        "steps": ["1. First step", "2. Second step", "3. Third step"],
        "application_contexts": ["context1", "context2"],
        "examples": ["example1", "example2"],
        "frequency_score": 8,
        "confidence_score": 0.9,
        "batch_evidence": ["supporting quote 1", "supporting quote 2"]
    }}
]

Only include frameworks that are clearly explained. Be generous in this MAP phase - capture all potential models.
Confidence score should be 0.6-1.0 based on clarity in this batch.
"""

        # Map phase prompt for core beliefs
        map_core_beliefs_prompt = """
You are analyzing a batch of content from an influencer to identify CORE BELIEFS and principles.

CONTENT BATCH:
{content}

STATISTICAL INSIGHTS:
{statistical_insights}

Your task is to extract CANDIDATE core beliefs from this batch. Look for:
1. Repeated philosophical statements
2. Fundamental principles about life/business/success
3. Strong opinions stated as facts
4. Values consistently promoted
5. Beliefs that underpin advice given

For each belief found, provide:
- STATEMENT: Clear, concise belief statement
- CATEGORY: Area it relates to (productivity, business, personal development, etc.)
- EVIDENCE: Supporting quotes/examples from this batch
- FREQUENCY_SCORE: How often this belief appears in this batch (1-10)
- CONFIDENCE: How sure you are this is a core belief

Return a JSON array of core beliefs:
[
    {{
        "statement": "Clear belief statement", 
        "category": "category_name",
        "evidence": ["supporting quote 1", "supporting quote 2"],
        "frequency_score": 7,
        "confidence_score": 0.8,
        "batch_source": "batch_description",
        "related_topics": ["topic1", "topic2"]
    }}
]

Be generous in this MAP phase - capture all potential beliefs that appear in this batch.
Focus on beliefs that are clearly stated or strongly implied in this specific content.
"""

        # Reduce phase prompt for mental models
        reduce_mental_models_prompt = """
You are consolidating mental models extracted from multiple batches of influencer content.

ALL CANDIDATE MENTAL MODELS FROM BATCHES:
{candidate_models}

CONSOLIDATION STRATEGY: {strategy}
FREQUENCY THRESHOLD: {min_frequency}
TARGET COUNT: {top_k}

Your task is to create the FINAL, consolidated list of mental models by:

1. **DEDUPLICATION**: Merge similar/duplicate models that represent the same concept
2. **FREQUENCY WEIGHTING**: Prioritize models that appeared across multiple batches
3. **QUALITY FILTERING**: Keep only the most clearly defined and useful models
4. **SYNTHESIS**: Combine evidence and examples from multiple batches

For each final mental model, provide:
- Synthesized information from all relevant batch candidates
- Combined evidence and examples
- Total frequency score across all batches
- Confidence based on consistency across batches

Return a JSON array of the top {top_k} consolidated mental models:
[
    {{
        "name": "Final Framework Name",
        "description": "Synthesized description",
        "steps": ["consolidated steps"],
        "application_contexts": ["all contexts found"],
        "examples": ["best examples from all batches"],
        "total_frequency": 15,
        "confidence_score": 0.95,
        "batch_sources": ["batch1", "batch2", "batch3"],
        "consolidation_notes": "How this was synthesized"
    }}
]

Focus on models that:
- Appeared in {min_frequency}+ batches OR had high frequency in fewer batches
- Are clearly actionable and well-defined
- Represent distinct, valuable frameworks
"""

        # Reduce phase prompt for core beliefs
        reduce_core_beliefs_prompt = """
You are consolidating core beliefs extracted from multiple batches of influencer content.

ALL CANDIDATE CORE BELIEFS FROM BATCHES:
{candidate_beliefs}

CONSOLIDATION STRATEGY: {strategy}
FREQUENCY THRESHOLD: {min_frequency}
TARGET COUNT: {top_k}

Your task is to create the FINAL, consolidated list of core beliefs by:

1. **DEDUPLICATION**: Merge similar beliefs that express the same principle
2. **FREQUENCY WEIGHTING**: Prioritize beliefs that appeared across multiple batches
3. **QUALITY FILTERING**: Keep only the most fundamental and clearly expressed beliefs
4. **SYNTHESIS**: Combine evidence from multiple batches

For each final belief, provide:
- Synthesized statement representing the core principle
- Combined evidence from all relevant batches
- Total frequency across all batches
- Confidence based on consistency

Return a JSON array of the top {top_k} consolidated core beliefs:
[
    {{
        "statement": "Final belief statement",
        "category": "primary_category",
        "evidence": ["best evidence from all batches"],
        "total_frequency": 12,
        "confidence_score": 0.9,
        "batch_sources": ["batch1", "batch2"],
        "related_mental_models": ["related frameworks"],
        "consolidation_notes": "How this belief was synthesized"
    }}
]

Focus on beliefs that:
- Appeared in {min_frequency}+ batches OR had high confidence in fewer batches
- Are fundamental to the influencer's worldview
- Are clearly actionable or prescriptive
- Represent distinct principles
"""

        return {
            "map_mental_models": map_mental_models_prompt,
            "map_core_beliefs": map_core_beliefs_prompt,
            "reduce_mental_models": reduce_mental_models_prompt,
            "reduce_core_beliefs": reduce_core_beliefs_prompt
        }
    
    def _batch_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Intelligently batch documents based on token limits and batch size
        
        Args:
            documents: List of documents to batch
            
        Returns:
            List of document batches
        """
        config = self.settings.map_reduce_extraction
        batch_size = config.batch_size
        max_tokens_per_batch = config.max_tokens_per_batch
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for doc in documents:
            content = doc.get('content', '')
            # Rough token estimation: 1 token ≈ 4 characters
            doc_tokens = len(content) // 4
            
            # Check if adding this document would exceed limits
            if (len(current_batch) >= batch_size or 
                current_tokens + doc_tokens > max_tokens_per_batch):
                
                if current_batch:  # Save current batch if not empty
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(doc)
            current_tokens += doc_tokens
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        self.logger.info(f"Created {len(batches)} batches from {len(documents)} documents")
        return batches
    
    async def _process_batch(self, batch_documents: List[Dict[str, Any]], 
                           batch_index: int, extraction_type: str,
                           statistical_insights: str) -> List[Union[MentalModel, CoreBelief]]:
        """
        Process a single batch of documents for extraction
        
        Args:
            batch_documents: Documents in this batch
            batch_index: Index of this batch
            extraction_type: 'mental_models' or 'core_beliefs'
            statistical_insights: Statistical insights about the full corpus
            
        Returns:
            List of extracted models or beliefs
        """
        # Check cache first
        cached_results = self.cache_manager.load_batch_result(batch_documents, extraction_type)
        if cached_results is not None:
            self.processing_stats["cached_batches"] += 1
            self.logger.debug(f"Loaded batch {batch_index} from cache")
            return cached_results
        
        # Prepare content for this batch
        batch_content = "\n\n".join([
            f"Document: {doc.get('source', f'doc_{i}')}\n{doc.get('content', '')}"
            for i, doc in enumerate(batch_documents)
        ])
        
        # Select appropriate prompt and LLM
        prompt_key = f"map_{extraction_type}"
        prompt = self.prompts[prompt_key].format(
            content=batch_content,
            statistical_insights=statistical_insights
        )
        
        try:
            # Process with LLM
            start_time = time.time()
            response = await self.map_llm.agenerate([[HumanMessage(content=prompt)]])
            processing_time = time.time() - start_time
            
            result_text = response.generations[0][0].text
            result_json = json.loads(result_text)
            
            # Convert to appropriate objects
            results = []
            for item_data in result_json:
                try:
                    if extraction_type == "mental_models":
                        results.append(MentalModel(
                            name=item_data.get('name', ''),
                            description=item_data.get('description', ''),
                            steps=item_data.get('steps', []),
                            application_contexts=item_data.get('application_contexts', []),
                            examples=item_data.get('examples', []),
                            confidence_score=item_data.get('confidence_score', 0.5)
                        ))
                    elif extraction_type == "core_beliefs":
                        results.append(CoreBelief(
                            statement=item_data.get('statement', ''),
                            category=item_data.get('category', ''),
                            evidence=item_data.get('evidence', []),
                            frequency=item_data.get('frequency_score', 1),
                            confidence_score=item_data.get('confidence_score', 0.5),
                            related_mental_models=item_data.get('related_topics', [])
                        ))
                except Exception as e:
                    self.logger.warning(f"Failed to create {extraction_type} object: {e}")
                    continue
            
            # Cache the results
            self.cache_manager.save_batch_result(
                batch_documents, extraction_type, results, 
                statistical_insights, batch_index
            )
            
            self.processing_stats["completed_batches"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            self.logger.debug(f"Processed batch {batch_index}: {len(results)} {extraction_type} in {processing_time:.1f}s")
            return results
            
        except Exception as e:
            self.processing_stats["failed_batches"] += 1
            self.logger.error(f"Failed to process batch {batch_index}: {e}")
            return []
    
    async def _map_phase(self, documents: List[Dict[str, Any]], 
                        extraction_type: str, 
                        statistical_insights: str) -> List[Union[MentalModel, CoreBelief]]:
        """
        Execute the map phase: process all document batches
        
        Args:
            documents: All documents to process
            extraction_type: 'mental_models' or 'core_beliefs'
            statistical_insights: Statistical insights about the corpus
            
        Returns:
            List of all candidate results from all batches
        """
        self.logger.info(f"Starting map phase for {extraction_type}")
        
        # Create batches
        batches = self._batch_documents(documents)
        self.processing_stats["total_batches"] = len(batches)
        
        # Check for existing consolidated results first
        consolidated_results = self.cache_manager.load_consolidated_result(documents, extraction_type)
        if consolidated_results is not None:
            self.logger.info(f"Found cached consolidated {extraction_type} results")
            return consolidated_results
        
        # Process batches with progress tracking
        all_candidates = []
        config = self.settings.map_reduce_extraction
        
        if config.show_progress:
            pbar = tqdm(total=len(batches), desc=f"Map phase ({extraction_type})", unit="batch")
        
        # Process batches in parallel groups
        for i in range(0, len(batches), config.parallel_batches):
            batch_group = batches[i:i + config.parallel_batches]
            
            # Create tasks for this group
            tasks = [
                self._process_batch(batch, i + j, extraction_type, statistical_insights)
                for j, batch in enumerate(batch_group)
            ]
            
            # Execute batch group in parallel
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Batch {i + j} failed: {result}")
                        continue
                    
                    all_candidates.extend(result)
                    
                    if config.show_progress:
                        pbar.update(1)
                        
            except Exception as e:
                self.logger.error(f"Failed to process batch group starting at {i}: {e}")
        
        if config.show_progress:
            pbar.close()
        
        self.logger.info(f"Map phase completed: {len(all_candidates)} candidates from {len(batches)} batches")
        return all_candidates
    
    async def _reduce_phase(self, candidates: List[Union[MentalModel, CoreBelief]], 
                          extraction_type: str,
                          all_documents: List[Dict[str, Any]]) -> List[Union[MentalModel, CoreBelief]]:
        """
        Execute the reduce phase: consolidate candidates into final results
        
        Args:
            candidates: All candidate results from map phase
            extraction_type: 'mental_models' or 'core_beliefs'
            all_documents: All original documents (for caching)
            
        Returns:
            List of consolidated final results
        """
        self.logger.info(f"Starting reduce phase for {extraction_type}")
        
        if not candidates:
            self.logger.warning(f"No candidates for {extraction_type} reduce phase")
            return []
        
        # Prepare candidates data for LLM
        candidates_data = []
        for candidate in candidates:
            if hasattr(candidate, 'dict'):
                candidates_data.append(candidate.dict())
            else:
                candidates_data.append(candidate)
        
        # Get consolidation configuration
        config = self.settings.map_reduce_extraction
        consolidation_config = config.mental_models if extraction_type == "mental_models" else config.core_beliefs
        
        strategy = consolidation_config.get('consolidation_strategy', 'frequency_weighted')
        min_frequency = consolidation_config.get('min_frequency', 2)
        top_k = consolidation_config.get('top_k', 50 if extraction_type == "mental_models" else 100)
        
        # Prepare consolidation prompt
        prompt_key = f"reduce_{extraction_type}"
        prompt = self.prompts[prompt_key].format(
            candidate_models=json.dumps(candidates_data, indent=2) if extraction_type == "mental_models" else json.dumps(candidates_data, indent=2),
            candidate_beliefs=json.dumps(candidates_data, indent=2) if extraction_type == "core_beliefs" else "",
            strategy=strategy,
            min_frequency=min_frequency,
            top_k=top_k
        )
        
        try:
            # Process with reduce LLM
            start_time = time.time()
            response = await self.reduce_llm.agenerate([[HumanMessage(content=prompt)]])
            processing_time = time.time() - start_time
            
            result_text = response.generations[0][0].text
            result_json = json.loads(result_text)
            
            # Convert to appropriate objects
            consolidated_results = []
            for item_data in result_json:
                try:
                    if extraction_type == "mental_models":
                        consolidated_results.append(MentalModel(
                            name=item_data.get('name', ''),
                            description=item_data.get('description', ''),
                            steps=item_data.get('steps', []),
                            application_contexts=item_data.get('application_contexts', []),
                            examples=item_data.get('examples', []),
                            confidence_score=item_data.get('confidence_score', 0.5)
                        ))
                    elif extraction_type == "core_beliefs":
                        consolidated_results.append(CoreBelief(
                            statement=item_data.get('statement', ''),
                            category=item_data.get('category', ''),
                            evidence=item_data.get('evidence', []),
                            frequency=item_data.get('total_frequency', 1),
                            confidence_score=item_data.get('confidence_score', 0.5),
                            related_mental_models=item_data.get('related_mental_models', [])
                        ))
                except Exception as e:
                    self.logger.warning(f"Failed to create consolidated {extraction_type} object: {e}")
                    continue
            
            # Cache consolidated results
            consolidation_metadata = {
                "strategy": strategy,
                "min_frequency": min_frequency,
                "top_k": top_k,
                "input_candidates": len(candidates),
                "output_results": len(consolidated_results),
                "processing_time": processing_time
            }
            
            self.cache_manager.save_consolidated_result(
                all_documents, extraction_type, consolidated_results, consolidation_metadata
            )
            
            self.logger.info(f"Reduce phase completed: {len(consolidated_results)} final {extraction_type} from {len(candidates)} candidates")
            return consolidated_results
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate {extraction_type}: {e}")
            return []
    
    async def extract_mental_models(self, documents: List[Dict[str, Any]], 
                                  statistical_insights: str) -> List[MentalModel]:
        """
        Extract mental models using map-reduce strategy
        
        Args:
            documents: All documents to analyze
            statistical_insights: Statistical insights about the corpus
            
        Returns:
            List of consolidated mental models
        """
        # Map phase: extract candidates from batches
        candidates = await self._map_phase(documents, "mental_models", statistical_insights)
        
        # Reduce phase: consolidate into final results
        final_results = await self._reduce_phase(candidates, "mental_models", documents)
        
        return final_results
    
    async def extract_core_beliefs(self, documents: List[Dict[str, Any]], 
                                 statistical_insights: str) -> List[CoreBelief]:
        """
        Extract core beliefs using map-reduce strategy
        
        Args:
            documents: All documents to analyze
            statistical_insights: Statistical insights about the corpus
            
        Returns:
            List of consolidated core beliefs
        """
        # Map phase: extract candidates from batches
        candidates = await self._map_phase(documents, "core_beliefs", statistical_insights)
        
        # Reduce phase: consolidate into final results
        final_results = await self._reduce_phase(candidates, "core_beliefs", documents)
        
        return final_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats["total_batches"] > 0:
            stats["completion_rate"] = stats["completed_batches"] / stats["total_batches"]
            stats["cache_hit_rate"] = stats["cached_batches"] / stats["total_batches"]
            stats["failure_rate"] = stats["failed_batches"] / stats["total_batches"]
        
        if stats["completed_batches"] > 0:
            stats["avg_processing_time_per_batch"] = stats["total_processing_time"] / stats["completed_batches"]
        
        return stats