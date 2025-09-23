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
from ..utils.llm_utils import robust_json_loads, clean_llm_json_response, clean_reduce_phase_json_response
from llm_output_parser import parse_json


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
                "max_tokens": 52428,  # 80% of Gemini 2.5 Flash max output (65536 tokens)
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
            self.logger.debug(f"LLM init error details: {str(e)}")
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
                "max_tokens": 52428,  # 80% of Gemini 2.5 Flash max output (65536 tokens)
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
            self.logger.debug(f"Reduce LLM init error details: {str(e)}")
            raise
    
    def _init_prompts(self) -> Dict[str, str]:
        """Initialize extraction prompts for map and reduce phases"""
        
        # Map phase prompt for mental models
        map_mental_models_prompt = """
You are a highly precise AI analyst acting as a knowledge extractor. Your task is to identify and structure the repeatable, actionable frameworks ("Mental Models") an influencer teaches by following a strict input-output format.

## TASK DEFINITION & GOAL ##

Your primary goal is to find and formalize structured, step-by-step processes from the provided text.

**What Qualifies as a "Mental Model":**
A Mental Model is a repeatable recipe or process for achieving a specific outcome. You must look for:
- **Explicitly named frameworks** (e.g., "The 5-Step Content Funnel," "My Audience Growth Flywheel").
- **Numbered or sequential instructions** (e.g., "There are three things you must do. First, you need to X. Second, you do Y...").
- **Clear, multi-step processes** for a specific task (e.g., validating a business idea, hiring a key employee).

**What to AVOID (These are NOT Mental Models):**
- **Vague advice:** Do not extract simple platitudes like "be consistent," "work hard," or "listen to your customers."
- **Single opinions or recommendations:** Do not extract statements like "I think Notion is the best tool" or "You should post on Twitter."
- **Philosophical statements:** Do not extract high-level beliefs that lack a clear, actionable sequence of steps.

You will receive an <input_block> with the content to analyze. You MUST generate an <output_block> that contains your detailed reasoning and the final, precisely formatted JSON.

---
## JSON FIELD DEFINITIONS ##

You must adhere to the following definitions for each field in the JSON object:

- **`name` (string):** A concise, descriptive name for the framework. If the influencer gives it a name, use that. If not, create a name that summarizes its purpose (e.g., "Startup Idea Validation Process").
- **`description` (string):** A single sentence explaining the purpose and outcome of this model. It must answer: "What does this framework help someone achieve?"
- **`steps` (array of strings):** An array where each string is a distinct, actionable step in the process, kept in the correct sequence.
- **`application_contexts` (array of strings):** A list of specific situations or domains where this model is applied (e.g., "Validating a startup idea", "Growing a YouTube channel").
- **`examples` (array of strings):** A list of brief, concrete examples of the model being used (e.g., "My last SaaS launch", "The 'Creator Funnel' campaign").
- **`frequency_score` (integer):** An estimated integer from 1 to 10 representing how often this model is mentioned *within this specific content batch*. A single clear mention is a 5; a central, repeated theme is a 9 or 10.
- **`confidence_score` (float):** A float between 0.0 and 1.0 representing your confidence that this is a true, well-defined model. Use 0.9+ for explicitly named frameworks with clear steps.
- **`batch_evidence` (array of strings):** A list containing 1-2 of the most compelling and direct verbatim quotes from the text that prove the existence of the model.

---
## EXAMPLE ##

<input_block>
<content_to_analyze>
To launch a new product, I always follow my 3-P Framework. First, you have to Plan your priorities. Figure out the one thing that matters. Second, Protect your time fiercely. Block it out on your calendar. Finally, Perform with focus. Turn off all distractions and just execute. I used this for my last SaaS launch and it was a game changer.
</content_to_analyze>
</input_block>

<output_block>
<thinking>
The user mentioned a "3-P Framework". It has three clear, sequential steps. This qualifies as a mental model. I will extract its name, description, steps, and context. It was used for a 'SaaS launch', so that is both the context and an example. It's a central theme here, so frequency is high. Confidence is very high as it's named and has clear steps. I will pull direct quotes for evidence.
</thinking>
<json_output>
[
    {{
        "name": "The 3-P Framework",
        "description": "A three-step framework for successfully launching a new product by focusing on priorities, time management, and execution.",
        "steps": [
            "1. Plan your priorities and identify the single most important task.",
            "2. Protect your time by blocking it out on a calendar.",
            "3. Perform with focus by eliminating distractions during execution."
        ],
        "application_contexts": ["Product Launches", "SaaS Business"],
        "examples": ["Used for my last SaaS launch"],
        "frequency_score": 8,
        "confidence_score": 0.95,
        "batch_evidence": ["To launch a new product, I always follow my 3-P Framework.", "Finally, Perform with focus. Turn off all distractions and just execute."]
    }}
]
</json_output>
</output_block>

---
## YOUR TASK ##

<input_block>
<content_to_analyze>
{content}
</content_to_analyze>
</input_block>

<output_block>"""

        # Map phase prompt for core beliefs
        map_core_beliefs_prompt = """
You are a highly precise AI analyst acting as a knowledge extractor. Your task is to identify and formalize the foundational principles, or "Core Beliefs," that guide an influencer's worldview and advice by following a strict input-output format.

## TASK DEFINITION & GOAL ##

Your primary goal is to uncover the fundamental, often repeated, rules and assumptions that underpin the influencer's content.

**What Qualifies as a "Core Belief":**
A Core Belief is a foundational principle that is treated as a truth. It's the "why" behind their advice. You must look for:
- **Underlying assumptions** (e.g., "Every business must leverage automation to survive").
- **Strong, principled stances** (e.g., "Action produces information; it's better to launch than to plan perfectly").
- **Frequently repeated philosophical rules** about how the world works in their domain.

**What to AVOID (These are NOT Core Beliefs):**
- **Simple opinions or preferences:** Do not extract statements like "I think AI is cool" or "Our next guest is fantastic."
- **Factual statements:** Do not extract verifiable facts like "The market is growing by 10% a year."
- **Specific, tactical advice:** Do not extract instructions that aren't tied to a deeper principle (e.g., "You should post three times a day on Twitter").

You will receive an <input_block> with the content to analyze. You MUST generate an <output_block> that contains your detailed reasoning and the final, precisely formatted JSON.

---
## JSON FIELD DEFINITIONS ##

You must adhere to the following definitions for each field in the JSON object:

- **`statement` (string):** A single, well-articulated sentence that captures the essence of the core belief. It should be a timeless principle, not specific advice.
- **`category` (string):** A single, lower-case keyword that best describes the domain of this belief (e.g., 'entrepreneurship', 'marketing', 'personal_development', 'mindset').
- **`evidence` (array of strings):** A list containing 1-2 of the most powerful, direct, verbatim quotes from the text that prove the influencer holds this belief.
- **`frequency_score` (integer):** An estimated integer from 1 to 10. How prominent is this belief *within this specific content batch*? A single, powerful statement might be a 6; a belief that is the central theme of the text would be a 9 or 10.
- **`confidence_score` (float):** A float between 0.0 and 1.0. How confident are you that this is a foundational, core belief and not just a fleeting opinion? Use 0.9+ for beliefs that are stated explicitly, with strong conviction, and are central to the main argument.
- **`batch_source` (string):** The identifier for the source of this content, which will be provided in the input. Use the value from the `{batch_description}` variable.
- **`related_topics` (array of strings):** A list of 2-3 keywords or short phrases that are closely related to this belief (e.g., for a belief about 'action over planning', related topics could be 'lean startup', 'prototyping', 'validation').

---
## EXAMPLE ##

<input_block>
<content_to_analyze>
Look, a lot of people get stuck in analysis paralysis. They plan forever. I believe you have to ship. Action produces information. You learn more from a failed launch than from a perfect plan that never sees the light of day. Just get it out there. It's the core of the lean startup methodology.
</content_to_analyze>
<batch_description>how_to_build_a_startup.txt</batch_description>
</input_block>

<output_block>
<thinking>
The influencer repeatedly emphasizes action over planning. The phrase "Action produces information" is a very strong, concise statement of this principle. This is a clear core belief. I will categorize it as 'entrepreneurship'. I will extract the most powerful quote as evidence. This is the central theme of the provided text, so frequency is high. Confidence is also very high because it's stated as a core belief. The related topics are about shipping products and validating ideas. The batch source is provided in the input.
</thinking>
<json_output>
[
    {{
        "statement": "Action produces information; it is better to launch and learn than to wait for a perfect plan.",
        "category": "entrepreneurship",
        "evidence": ["You learn more from a failed launch than from a perfect plan that never sees the light of day."],
        "frequency_score": 9,
        "confidence_score": 0.95,
        "batch_source": "how_to_build_a_startup.txt",
        "related_topics": ["lean startup", "product validation", "execution"]
    }}
]
</json_output>
</output_block>

---
## YOUR TASK ##

<input_block>
<content_to_analyze>
{content}
</content_to_analyze>
<batch_description>{batch_description}</batch_description>
</input_block>

<output_block>"""

        # Reduce phase prompt for mental models
        reduce_mental_models_prompt = """
You are a senior AI strategist. Your task is to analyze a large collection of candidate "Mental Models" and produce a final, consolidated, and de-duplicated list of the top {top_k} frameworks, strictly adhering to the provided instructions and the required JSON format.

## TASK DEFINITION & GOAL ##

Your primary goal is to synthesize the raw data from the <candidate_models_json> into a clean, definitive, and ranked list of the most important mental models. You will achieve this by following a precise algorithm.

**Consolidation Algorithm:**
1.  **Analyze & Cluster:** First, carefully review all candidate models provided. Group them into clusters where each model in a cluster represents the same core framework, even if the names or details are slightly different.
2.  **Synthesize Each Cluster:** For each cluster of candidate models, you must create one single, master version. Merge the descriptions into a single, clear statement. Combine and refine the `steps` into a single, logical, and de-duplicated sequence.
3.  **Aggregate Data:** Combine all unique `application_contexts` and `examples` from the cluster into unified lists.
4.  **Calculate Final Scores:** For each consolidated model, calculate the `total_frequency` by SUMMING the `frequency_score` of all candidates in its cluster. Calculate the final `confidence_score` by taking the AVERAGE of the `confidence_score` of all candidates in its cluster.
5.  **Filter & Rank:** Apply the filtering logic based on the `CONSOLIDATION STRATEGY: {strategy}` and `FREQUENCY THRESHOLD: {min_frequency}`. Then, rank the resulting models and select the `TARGET COUNT: {top_k}`.
6.  **Format Output:** Format the final ranked list into the required JSON structure, providing all required fields.

---
## JSON FIELD DEFINITIONS ##

You must generate a final JSON ARRAY containing multiple consolidated models. Each object in the array must have ONLY the following fields:

- **`name` (string):** The final, most common or clearest name for the framework from the cluster.
- **`description` (string):** A comprehensive, synthesized description that combines the best elements from all candidates in the cluster.
- **`steps` (array of strings):** A de-duplicated and logically ordered list of steps that represents the complete framework.
- **`application_contexts` (array of strings):** A combined, de-duplicated array of all unique contexts from the cluster.
- **`examples` (array of strings):** A combined, de-duplicated array of the best and most illustrative examples from the cluster.
- **`total_frequency` (integer):** The SUM of all `frequency_score` values from the original candidates that were merged into this final model.
- **`confidence_score` (float):** The AVERAGE of all `confidence_score` values from the merged candidates, rounded to two decimal places.
- **`batch_sources` (array of strings):** An array of strings listing the `batch_source` of *every* candidate that was merged into this final model.
- **`consolidation_notes` (string):** A brief, one-sentence explanation of how the model was consolidated (e.g., "Merged 3 candidates related to the '3-P Framework'").

---
## EXAMPLE ##

<input_block>
<candidate_models_json>
[
    {{"name": "The 3-P Launch Method", "description": "A framework for launching products.", "steps": ["1. Plan priorities.", "2. Protect time.", "3. Perform with focus."], "application_contexts": ["Product Launches"], "examples": ["SaaS launch"], "frequency_score": 8, "confidence_score": 0.95, "batch_source": "batch_001.txt"}},
    {{"name": "My 3-P Framework", "description": "How to execute on a project.", "steps": ["1. Plan the work.", "2. Protect your calendar.", "3. Perform the execution."], "application_contexts": ["Project Execution"], "examples": ["My last big project"], "frequency_score": 6, "confidence_score": 0.90, "batch_source": "batch_005.txt"}}
]
</candidate_models_json>
</input_block>

<output_block>
<thinking>
The two candidate models clearly refer to the same "3-P Framework". I will merge them into a single cluster. I will synthesize the name to "The 3-P Framework" and combine the descriptions. The steps are similar, so I will merge and refine them. I will create a unique list of contexts and examples. I will calculate the total_frequency by summing 8 + 6 = 14. I will calculate the average confidence score as (0.95 + 0.90) / 2 = 0.925, which I'll round to 0.93. I will list both batch sources.
</thinking>
<json_output>
[
    {{
        "name": "The 3-P Framework",
        "description": "A three-step framework for launching products and executing on projects by planning priorities, protecting time, and performing with focus.",
        "steps": ["1. Plan priorities and the work.", "2. Protect time on your calendar.", "3. Perform with focused execution."],
        "application_contexts": ["Product Launches", "Project Execution"],
        "examples": ["SaaS launch", "My last big project"],
        "total_frequency": 14,
        "confidence_score": 0.93,
        "batch_sources": ["batch_001.txt", "batch_005.txt"],
        "consolidation_notes": "Merged 2 candidates representing the same 3-P framework."
    }}
]
</json_output>
</output_block>

---
## YOUR TASK ##

<input_block>
<candidate_models_json>
{candidate_models}
</candidate_models_json>
</input_block>

<output_block>

**IMPORTANT**: Your response must be a complete JSON array. Ensure your response ends with a proper closing bracket ']' to form valid JSON."""

        # Reduce phase prompt for core beliefs
        reduce_core_beliefs_prompt = """
You are a senior AI strategist and philosopher. Your task is to analyze a large collection of candidate "Core Beliefs" and produce a final, consolidated, and de-duplicated list of the top {top_k} foundational principles, strictly adhering to the provided instructions and the required JSON format.

## TASK DEFINITION & GOAL ##

Your primary goal is to synthesize the raw data from the <candidate_beliefs_json> into a clean, definitive, and ranked list of the most important core beliefs. You will achieve this by following a precise algorithm.

**Consolidation Algorithm:**
1.  **Analyze & Cluster:** First, carefully review all candidate beliefs provided. Group them into clusters where each belief in a cluster represents the same underlying principle, even if the wording of the `statement` is different.
2.  **Synthesize Each Cluster:** For each cluster, create one single, master belief. Your most important task is to write a new, elegant `statement` that captures the core idea of all candidates in the group.
3.  **Aggregate Data:** Combine all unique `evidence` quotes into a master list and select the most powerful 1-2 quotes. De-duplicate all `related_topics`.
4.  **Calculate Final Scores:** For each consolidated belief, calculate the `total_frequency` by SUMMING the `frequency_score` of all candidates in its cluster. Calculate the final `confidence_score` by taking the AVERAGE of the `confidence_score` of all candidates in its cluster.
5.  **Filter & Rank:** Apply the filtering logic based on the `CONSOLIDATION STRATEGY: {strategy}` and `FREQUENCY THRESHOLD: {min_frequency}`. Then, rank the resulting beliefs and select the `TARGET COUNT: {top_k}`.
6.  **Format Output:** Format the final ranked list into the required JSON structure, providing all required fields.

---
## JSON FIELD DEFINITIONS ##

You must generate a final JSON ARRAY containing multiple consolidated beliefs. Each object in the array must have ONLY the following fields:

- **`statement` (string):** The final, master statement of the belief, synthesized to be as clear and profound as possible.
- **`category` (string):** The single, most fitting category for the synthesized belief from the merged group.
- **`evidence` (array of strings):** An array containing the BEST 1-2 verbatim quotes from across all merged candidates.
- **`total_frequency` (integer):** The SUM of all `frequency_score` values from the original candidates that were merged into this final belief.
- **`confidence_score` (float):** The AVERAGE of all `confidence_score` values from the merged candidates, rounded to two decimal places.
- **`batch_sources` (array of strings):** An array of strings listing the `batch_source` of *every* candidate that was merged into this final belief.
- **`related_mental_models` (array of strings):** A list of names of any mental models that appear to be directly related to this core belief, based on the evidence and topics. If none, provide an empty array.
- **`consolidation_notes` (string):** A brief, one-sentence explanation of the consolidation (e.g., "Merged 2 candidates about the importance of action over planning.").

---
## EXAMPLE ##

<input_block>
<candidate_beliefs_json>
[
    {{"statement": "You must act to get information.", "category": "entrepreneurship", "evidence": ["Action produces information..."], "frequency_score": 9, "confidence_score": 0.95, "batch_source": "batch_001.txt", "related_topics": ["lean startup", "validation"]}},
    {{"statement": "Launching is better than perfect planning.", "category": "entrepreneurship", "evidence": ["You learn more from a failed launch than a perfect plan..."], "frequency_score": 7, "confidence_score": 0.90, "batch_source": "batch_002.txt", "related_topics": ["prototyping", "execution"]}}
]
</candidate_beliefs_json>
</input_block>

<output_block>
<thinking>
The two candidate beliefs express the same core idea of valuing action over planning. I will merge them into a single cluster. I will synthesize a new, more comprehensive statement. The category is consistent. I will select the best evidence from both. I will sum the frequencies (9 + 7 = 16) and average the confidences ((0.95 + 0.90) / 2 = 0.925, rounded to 0.93). I will list both batch sources. The topics are all related and can be merged. I do not see any specific mental models mentioned, so I will leave that field empty. I will write a consolidation note.
</thinking>
<json_output>
[
    {{
        "statement": "Action produces information; it is better to launch and learn than to wait for a perfect plan.",
        "category": "entrepreneurship",
        "evidence": ["Action produces information...", "You learn more from a failed launch than a perfect plan..."],
        "total_frequency": 16,
        "confidence_score": 0.93,
        "batch_sources": ["batch_001.txt", "batch_002.txt"],
        "related_mental_models": [],
        "consolidation_notes": "Merged 2 candidates about the principle of prioritizing action over planning."
    }}
]
</json_output>
</output_block>

---
## YOUR TASK ##

<input_block>
<candidate_beliefs_json>
{candidate_beliefs}
</candidate_beliefs_json>
</input_block>

<output_block>

**IMPORTANT**: Your response must be a complete JSON array. Ensure your response ends with a proper closing bracket ']' to form valid JSON."""

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
            self.logger.info(f"Loaded batch {batch_index} from cache")
            return cached_results
        
        # Calculate batch hash for logging
        batch_hash = self.cache_manager._calculate_batch_hash(batch_documents, extraction_type)
        
        # Prepare content for this batch
        batch_content = "\n\n".join([
            f"Document: {doc.get('source', f'doc_{i}')}\n{doc.get('content', '')}"
            for i, doc in enumerate(batch_documents)
        ])
        
        # Select appropriate prompt and LLM
        prompt_key = f"map_{extraction_type}"
        
        if extraction_type == "mental_models":
            # Mental models prompt only needs content
            prompt = self.prompts[prompt_key].format(
                content=batch_content
            )
        else:
            # Core beliefs uses batch_description instead of statistical_insights
            # Generate batch description from document sources
            batch_description = f"batch_{batch_index}_" + "_".join([
                doc.get('source', f'doc_{i}')[:15] for i, doc in enumerate(batch_documents[:3])
            ])
            prompt = self.prompts[prompt_key].format(
                content=batch_content,
                batch_description=batch_description
            )
        
        # Create batch log directory and save input
        batch_log_dir = self.cache_manager.create_batch_log_directory(batch_hash, extraction_type)
        self.cache_manager.save_batch_input(batch_log_dir, prompt)
        
        # Debug logging for prompt
        self.logger.debug(f"Processing batch {batch_index} with {len(batch_documents)} documents")
        
        try:
            # Process with LLM
            start_time = time.time()
            response = await self.map_llm.agenerate([[HumanMessage(content=prompt)]])
            processing_time = time.time() - start_time
            self.logger.debug(f"LLM response received in {processing_time:.2f}s for batch {batch_index}")
            
            result_text = response.generations[0][0].text
            
            # Debug logging for response
            if not result_text:
                self.logger.error(f"Empty/None response from LLM for batch {batch_index} (hash: {batch_hash[:12]})")
            
            # Save complete response
            self.cache_manager.save_batch_response(batch_log_dir, result_text)
            
            # Parse JSON - try llm-output-parser first (better with markdown/mixed content)
            try:
                # Try llm-output-parser first (handles markdown wrapped JSON better)
                try:
                    result_json = parse_json(result_text)
                    self.logger.debug(f"llm-output-parser successful, got {len(result_json) if isinstance(result_json, list) else 'non-list'} items")
                except Exception as parse_error:
                    self.logger.debug(f"llm-output-parser failed: {str(parse_error)}, falling back to robust_json_loads")
                    # Fallback to robust XML-aware extraction
                    result_json = robust_json_loads(result_text, self.logger)
                    self.logger.debug(f"robust_json_loads successful, got {len(result_json) if isinstance(result_json, list) else 'non-list'} items")
                
                # Check for truncation indicators
                if result_text and not result_text.strip().endswith((']', '}', '```', '</json_output>', '</output_block>')):
                    self.logger.warning(f"Response for batch {batch_index} appears truncated - does not end with proper closing")
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"JSON parse failed for batch {batch_index} (hash: {batch_hash[:12]}): {str(e)}")
                self.logger.error(f"Check batch logs at: xml_responses/{extraction_type}/batch_{batch_hash[:16]}/")
                self.logger.debug(f"Failed response text (first 1000 chars): {result_text[:1000] if result_text else 'None'}")
                raise
            
            # Save extracted JSON
            self.cache_manager.save_batch_output(batch_log_dir, result_json)
            
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
            
            # Save batch metadata
            batch_metadata = {
                "batch_index": batch_index,
                "batch_hash": batch_hash,
                "extraction_type": extraction_type,
                "document_count": len(batch_documents),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.settings.map_reduce_extraction.map_phase_model,
                "result_count": len(results),
                "document_sources": [doc.get('source', f'doc_{i}') for i, doc in enumerate(batch_documents)]
            }
            self.cache_manager.save_batch_metadata(batch_log_dir, batch_metadata)
            
            # Cache the results
            self.cache_manager.save_batch_result(
                batch_documents, extraction_type, results, 
                statistical_insights, batch_index
            )
            
            self.processing_stats["completed_batches"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            self.logger.debug(f"Processed batch {batch_index} (hash: {batch_hash[:12]}): {len(results)} {extraction_type} in {processing_time:.1f}s")
            return results
            
        except Exception as e:
            self.processing_stats["failed_batches"] += 1
            self.logger.error(f"Failed to process batch {batch_index} (hash: {batch_hash[:12]}): {e}")
            
            # Save error metadata
            error_metadata = {
                "batch_index": batch_index,
                "batch_hash": batch_hash,
                "extraction_type": extraction_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            try:
                self.cache_manager.save_batch_metadata(batch_log_dir, error_metadata)
            except:
                pass
            
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
        
        # Scan for cached batches to show resume status
        cached_batch_count = 0
        for i, batch in enumerate(batches):
            cached_result = self.cache_manager.load_batch_result(batch, extraction_type)
            if cached_result is not None:
                cached_batch_count += 1
        
        # Display resume status information
        if cached_batch_count > 0:
            self.logger.info(f"Resume detected - Found {cached_batch_count} cached batches from previous run ({cached_batch_count} of {len(batches)} batches)")
            self.logger.info(f"Will process {len(batches) - cached_batch_count} new batches and load {cached_batch_count} from cache")
        else:
            self.logger.info(f"Starting fresh extraction - Processing all {len(batches)} batches")
        
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
                        # Update progress bar with cache statistics
                        cached_so_far = self.processing_stats.get("cached_batches", 0)
                        processed_so_far = (i + j + 1) - cached_so_far
                        cache_rate = (cached_so_far / (i + j + 1)) * 100 if (i + j + 1) > 0 else 0
                        pbar.set_postfix({
                            'cached': cached_so_far,
                            'processed': processed_so_far,
                            'cache_rate': f"{cache_rate:.1f}%"
                        })
                        pbar.update(1)
                        
            except Exception as e:
                self.logger.error(f"Failed to process batch group starting at {i}: {e}")
        
        if config.show_progress:
            pbar.close()
        
        # Calculate and display final statistics
        cached_batches = self.processing_stats.get("cached_batches", 0)
        processed_batches = len(batches) - cached_batches
        cache_hit_rate = (cached_batches / len(batches)) * 100 if len(batches) > 0 else 0
        
        self.logger.info(f"Map phase completed: {len(all_candidates)} candidates from {len(batches)} batches")
        self.logger.info(f"Cache efficiency - {cached_batches} cached, {processed_batches} processed ({cache_hit_rate:.1f}% cache hit rate)")
        
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
        
        self.logger.debug(f"Reduce phase for {extraction_type} - consolidating {len(candidates_data)} candidates")
        
        # Create reduce log directory and save input
        reduce_log_dir = self.cache_manager.create_reduce_log_directory(extraction_type)
        self.cache_manager.save_reduce_input(reduce_log_dir, prompt, candidates_data)
        
        try:
            # Process with reduce LLM
            start_time = time.time()
            response = await self.reduce_llm.agenerate([[HumanMessage(content=prompt)]])
            processing_time = time.time() - start_time
            self.logger.debug(f"Reduce LLM response received in {processing_time:.2f}s")
            
            result_text = response.generations[0][0].text
            if not result_text:
                self.logger.error(f"Empty/None reduce response from LLM")
            
            # Save complete response
            self.cache_manager.save_reduce_response(reduce_log_dir, result_text)
            
            # Parse JSON using llm-output-parser library
            try:
                self.logger.info(f"DEBUG: Raw reduce response (first 500 chars): {result_text[:500] if result_text else 'None'}...")
                self.logger.info(f"DEBUG: Raw response total length: {len(result_text) if result_text else 0}")
                
                # Use specialized LLM output parser designed for markdown/mixed content
                try:
                    result_json = parse_json(result_text)
                    self.logger.info(f"DEBUG: llm-output-parser successful")
                except Exception as parse_error:
                    self.logger.error(f"DEBUG: llm-output-parser failed: {str(parse_error)}")
                    # Fallback to direct JSON parsing as last resort
                    try:
                        result_json = json.loads(result_text)
                        self.logger.info(f"DEBUG: Direct JSON parse successful as fallback")
                    except json.JSONDecodeError as fallback_error:
                        self.logger.error(f"DEBUG: All parsing methods failed. Original error: {str(parse_error)}, Fallback error: {str(fallback_error)}")
                        raise parse_error
                
                self.logger.info(f"DEBUG: Parsed result_json type: {type(result_json)}")
                self.logger.info(f"DEBUG: Result_json length: {len(result_json) if isinstance(result_json, (list, dict)) else 'not list/dict'}")
                
                # Check for truncated response indicators
                if result_text and not result_text.strip().endswith((']', '}', '```')):
                    self.logger.warning(f"Response appears truncated - does not end with proper JSON closing")
                    
                # Validate reasonable output count (warn if suspiciously low)
                expected_min_items = max(2, len(candidates) // 10)  # At least 10% consolidation
                if isinstance(result_json, list) and len(result_json) < expected_min_items:
                    self.logger.warning(f"Suspiciously low output count: {len(result_json)} items from {len(candidates)} candidates (expected at least {expected_min_items})")
                
                # Ensure result_json is a list for iteration
                if isinstance(result_json, dict):
                    self.logger.info(f"DEBUG: Converting single dict to list")
                    result_json = [result_json]
                elif not isinstance(result_json, list):
                    raise ValueError(f"Expected list or dict from JSON parsing, got {type(result_json)}")
                
                if isinstance(result_json, list) and result_json:
                    self.logger.info(f"DEBUG: First item type: {type(result_json[0])}")
                    self.logger.info(f"DEBUG: First item content: {str(result_json[0])[:200]}...")
                self.logger.debug(f"Reduce JSON parse successful, got {len(result_json) if isinstance(result_json, list) else 'non-list'} consolidated items")
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Reduce JSON parse failed for {extraction_type}: {str(e)}")
                self.logger.debug(f"Failed reduce response text (first 1000 chars): {result_text[:1000] if result_text else 'None'}")
                raise
            
            # Convert to appropriate objects
            consolidated_results = []
            for i, item_data in enumerate(result_json):
                try:
                    self.logger.info(f"DEBUG: Processing item {i}: type={type(item_data)}, content={str(item_data)[:100]}...")
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
                    self.logger.info(f"DEBUG: Exception processing item {i}: {e}")
                    self.logger.warning(f"Failed to create consolidated {extraction_type} object: {e}")
                    continue
            
            # Save parsed JSON output and metadata
            self.cache_manager.save_reduce_output(reduce_log_dir, result_json)
            
            # Prepare comprehensive reduce metadata
            reduce_metadata = {
                "extraction_type": extraction_type,
                "timestamp": datetime.now().isoformat(),
                "consolidation_strategy": strategy,
                "min_frequency": min_frequency,
                "top_k": top_k,
                "input_candidates": len(candidates),
                "output_results": len(consolidated_results),
                "processing_time": processing_time,
                "model_used": self.settings.map_reduce_extraction.reduce_phase_model,
                "parsing_method": "llm-output-parser",
                "prompt_length": len(prompt),
                "response_length": len(result_text) if result_text else 0,
                "parsed_json_length": len(result_json) if isinstance(result_json, list) else 1,
                "consolidation_ratio": len(consolidated_results) / len(candidates) if len(candidates) > 0 else 0
            }
            self.cache_manager.save_reduce_metadata(reduce_log_dir, reduce_metadata)
            
            # Cache consolidated results (existing functionality)
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
            
            # Save error metadata if reduce_log_dir was created
            try:
                if 'reduce_log_dir' in locals():
                    error_metadata = {
                        "extraction_type": extraction_type,
                        "timestamp": datetime.now().isoformat(),
                        "consolidation_strategy": strategy,
                        "min_frequency": min_frequency,
                        "top_k": top_k,
                        "input_candidates": len(candidates),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "processing_stage": "reduce_phase",
                        "model_used": self.settings.map_reduce_extraction.reduce_phase_model
                    }
                    self.cache_manager.save_reduce_metadata(reduce_log_dir, error_metadata)
            except Exception as meta_error:
                self.logger.warning(f"Failed to save error metadata: {meta_error}")
            
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