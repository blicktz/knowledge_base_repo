"""
LLM-based persona extractor for distilling influencer personality from content
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm

from ..data.models.persona_constitution import (
    PersonaConstitution,
    LinguisticStyle,
    MentalModel, 
    CoreBelief,
    StatisticalReport,
    ExtractionMetadata
)
from ..core.statistical_analyzer import StatisticalAnalyzer
from ..config.settings import Settings
from ..utils.logging import get_logger


class PersonaExtractor:
    """
    Extracts persona constitution from influencer content using LLM analysis
    combined with statistical insights.
    """
    
    def __init__(self, settings: Settings):
        """Initialize the persona extractor"""
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Initialize LLM
        self.llm = None
        self._init_llm()
        
        # Initialize statistical analyzer
        self.statistical_analyzer = StatisticalAnalyzer(settings)
        
        # Extraction prompts
        self.prompts = self._init_prompts()
        
        # Track extraction state
        self.extraction_start_time = None
        
    def _init_llm(self):
        """Initialize the LLM based on configuration"""
        llm_config = self.settings.get_llm_config()
        provider = self.settings.llm.provider
        
        try:
            if provider == "openrouter" or provider == "openai":
                self.llm = ChatOpenAI(
                    openai_api_key=llm_config.get('api_key'),
                    openai_api_base=llm_config.get('base_url', 'https://openrouter.ai/api/v1'),
                    model_name=llm_config.get('model', 'gpt-4'),
                    temperature=llm_config.get('temperature', 0.1),
                    max_tokens=llm_config.get('max_tokens', 4000),
                    timeout=llm_config.get('timeout', 60),
                    max_retries=llm_config.get('num_retries', 3)
                )
            elif provider == "anthropic":
                self.llm = ChatAnthropic(
                    anthropic_api_key=llm_config.get('api_key'),
                    model=llm_config.get('model', 'claude-3-sonnet-20240229'),
                    temperature=llm_config.get('temperature', 0.1),
                    max_tokens=llm_config.get('max_tokens', 4000),
                    timeout=llm_config.get('timeout', 60)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            self.logger.info(f"Initialized LLM: {provider} - {llm_config.get('model')}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize extraction prompts"""
        
        # Linguistic style extraction prompt
        linguistic_style_prompt = PromptTemplate(
            input_variables=["content", "statistical_insights"],
            template="""
You are an expert linguist analyzing the speaking style of an influencer from their content.

CONTENT TO ANALYZE:
{content}

STATISTICAL INSIGHTS:
{statistical_insights}

Your task is to extract the influencer's linguistic style. Focus on:

1. TONE: Overall communication tone (energetic, direct, conversational, etc.)
2. CATCHPHRASES: Signature phrases they use repeatedly 
3. VOCABULARY: Key terms and words they favor
4. SENTENCE STRUCTURES: Common sentence patterns
5. COMMUNICATION STYLE: Formality level, directness, use of examples, etc.

Return a JSON object with this exact structure:
{{
    "tone": "Brief description of overall tone",
    "catchphrases": ["phrase1", "phrase2", "phrase3"],
    "vocabulary": ["term1", "term2", "term3"], 
    "sentence_structures": ["pattern1", "pattern2"],
    "communication_style": {{
        "formality": "formal/informal/mixed",
        "directness": "very_direct/direct/indirect",
        "use_of_examples": "frequent/occasional/rare",
        "storytelling": "high/medium/low",
        "humor": "frequent/occasional/rare"
    }}
}}

Be specific and evidence-based. Extract actual phrases and patterns from the content.
"""
        )
        
        # Mental models extraction prompt
        mental_models_prompt = PromptTemplate(
            input_variables=["content", "statistical_insights"],
            template="""
You are an expert in cognitive frameworks analyzing an influencer's problem-solving approaches.

CONTENT TO ANALYZE:
{content}

STATISTICAL INSIGHTS:
{statistical_insights}

Your task is to identify MENTAL MODELS and FRAMEWORKS the influencer uses for problem-solving.

Look for:
1. Systematic approaches to problems
2. Step-by-step methodologies
3. Frameworks with clear stages
4. Problem-solving patterns
5. Decision-making processes

For each mental model, provide:
- NAME: Clear, descriptive name
- DESCRIPTION: What the model does
- STEPS: Numbered steps (2-10 steps)
- APPLICATION CONTEXTS: Where it's used
- EXAMPLES: Real examples from content

Return a JSON array of mental models:
[
    {{
        "name": "Framework Name",
        "description": "Brief description of what this framework does",
        "steps": ["1. First step", "2. Second step", "3. Third step"],
        "application_contexts": ["context1", "context2"],
        "examples": ["example1", "example2"],
        "confidence_score": 0.9
    }}
]

Only include frameworks that are clearly explained with steps. Minimum 2 steps, maximum 10 steps.
Confidence score should be 0.6-1.0 based on how clearly the framework is explained.
"""
        )
        
        # Core beliefs extraction prompt
        core_beliefs_prompt = PromptTemplate(
            input_variables=["content", "statistical_insights"],
            template="""
You are analyzing an influencer's fundamental beliefs and principles from their content.

CONTENT TO ANALYZE:
{content}

STATISTICAL INSIGHTS:
{statistical_insights}

Your task is to identify CORE BELIEFS - fundamental principles that guide the influencer's thinking.

Look for:
1. Repeated philosophical statements
2. Fundamental principles about life/business/success
3. Strong opinions stated as facts
4. Values they consistently promote
5. Beliefs that underpin their advice

For each belief, provide:
- STATEMENT: Clear, concise belief statement
- CATEGORY: Area it relates to (productivity, business, personal development, etc.)
- EVIDENCE: Supporting quotes/examples from content
- FREQUENCY: How often this belief appears
- CONFIDENCE: How sure you are this is a core belief

Return a JSON array of core beliefs:
[
    {{
        "statement": "Clear belief statement", 
        "category": "category_name",
        "evidence": ["supporting quote 1", "supporting quote 2"],
        "frequency": 5,
        "confidence_score": 0.8,
        "related_mental_models": ["framework1", "framework2"]
    }}
]

Focus on beliefs that are:
- Clearly stated or strongly implied
- Repeated or reinforced multiple times
- Fundamental to their worldview
- Actionable or prescriptive

Confidence score should be 0.6-1.0 based on how clearly and frequently the belief is expressed.
"""
        )
        
        return {
            "linguistic_style": linguistic_style_prompt,
            "mental_models": mental_models_prompt,
            "core_beliefs": core_beliefs_prompt
        }
    
    async def extract_persona(self, documents: List[Dict[str, Any]]) -> PersonaConstitution:
        """
        Extract complete persona constitution from documents
        
        Args:
            documents: List of document dictionaries with 'content' key
            
        Returns:
            PersonaConstitution object
        """
        self.extraction_start_time = time.time()
        
        # Estimate total processing time based on content size
        total_words = sum(len(doc.get('content', '').split()) for doc in documents)
        estimated_time = self._estimate_processing_time(total_words, len(documents))
        
        self.logger.info(f"Starting persona extraction from {len(documents)} documents ({total_words:,} words)")
        self.logger.info(f"Estimated processing time: {estimated_time:.1f} minutes")
        
        # Define extraction steps for progress tracking
        extraction_steps = [
            ("Statistical analysis", lambda: self.statistical_analyzer.analyze_content(documents)),
            ("Preparing content", lambda: (self._prepare_content(documents), self._format_statistical_insights(None))),
            ("Linguistic style", lambda: None),  # Placeholder, will be updated
            ("Mental models", lambda: None),     # Placeholder, will be updated  
            ("Core beliefs", lambda: None)      # Placeholder, will be updated
        ]
        
        # Initialize progress bar
        with tqdm(total=len(extraction_steps), desc="Persona Extraction", unit="step") as pbar:
            # Step 1: Statistical analysis
            pbar.set_description("Persona Extraction: Statistical analysis")
            statistical_report = self.statistical_analyzer.analyze_content(documents)
            pbar.update(1)
            
            # Step 2: Prepare content for LLM analysis
            pbar.set_description("Persona Extraction: Preparing content")
            combined_content = self._prepare_content(documents)
            statistical_insights = self._format_statistical_insights(statistical_report)
            pbar.update(1)
            
            # Step 3: Extract different persona components
            pbar.set_description("Persona Extraction: Linguistic style")
            linguistic_style = await self._extract_linguistic_style(combined_content, statistical_insights)
            pbar.update(1)
            
            pbar.set_description("Persona Extraction: Mental models")
            mental_models = await self._extract_mental_models(combined_content, statistical_insights)
            pbar.update(1)
            
            pbar.set_description("Persona Extraction: Core beliefs")
            core_beliefs = await self._extract_core_beliefs(combined_content, statistical_insights)
            pbar.update(1)
        
        # Step 4: Create extraction metadata
        extraction_metadata = self._create_extraction_metadata(documents)
        
        # Step 5: Extract additional characteristics
        expertise_domains = self._extract_expertise_domains(statistical_report)
        content_themes = self._extract_content_themes(statistical_report)
        communication_preferences = self._infer_communication_preferences(linguistic_style)
        
        # Step 6: Build complete persona constitution
        persona_constitution = PersonaConstitution(
            linguistic_style=linguistic_style,
            mental_models=mental_models,
            core_beliefs=core_beliefs,
            statistical_report=statistical_report,
            extraction_metadata=extraction_metadata,
            expertise_domains=expertise_domains,
            content_themes=content_themes,
            communication_preferences=communication_preferences
        )
        
        total_time = time.time() - self.extraction_start_time
        self.logger.info(f"Persona extraction completed in {total_time:.2f} seconds")
        
        return persona_constitution
    
    def _prepare_content(self, documents: List[Dict[str, Any]], max_tokens: int = 15000) -> str:
        """Prepare and truncate content for LLM analysis"""
        # Combine all content
        all_content = []
        
        for doc in documents:
            content = doc.get('content', '')
            if content.strip():
                all_content.append(content)
        
        combined = "\n\n".join(all_content)
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_tokens = len(combined) // 4
        
        if estimated_tokens > max_tokens:
            # Truncate to fit within token limit
            max_chars = max_tokens * 4
            combined = combined[:max_chars]
            self.logger.warning(f"Content truncated to {max_tokens} tokens for LLM analysis")
        
        return combined
    
    def _format_statistical_insights(self, report: StatisticalReport) -> str:
        """Format statistical insights for LLM consumption"""
        insights = []
        
        insights.append(f"Total documents: {report.total_documents}")
        insights.append(f"Total words: {report.total_words:,}")
        insights.append(f"Total sentences: {report.total_sentences:,}")
        
        if report.top_keywords:
            top_keywords = list(report.top_keywords.keys())[:10]
            insights.append(f"Top keywords: {', '.join(top_keywords)}")
        
        if report.top_entities:
            top_entities = list(report.top_entities.keys())[:10]
            insights.append(f"Top entities: {', '.join(top_entities)}")
        
        if report.top_collocations:
            top_collocations = [item.get('ngram', '') for item in report.top_collocations[:5]]
            insights.append(f"Top phrases: {', '.join(top_collocations)}")
        
        if report.readability_metrics:
            flesch = report.readability_metrics.get('flesch_reading_ease', 0)
            insights.append(f"Reading ease: {flesch}")
        
        if report.linguistic_patterns:
            question_ratio = report.linguistic_patterns.get('question_ratio', 0)
            exclamation_ratio = report.linguistic_patterns.get('exclamation_ratio', 0)
            insights.append(f"Questions: {question_ratio:.1%}, Exclamations: {exclamation_ratio:.1%}")
        
        return "\n".join(insights)
    
    async def _extract_linguistic_style(self, content: str, statistical_insights: str) -> LinguisticStyle:
        """Extract linguistic style using LLM"""
        
        prompt = self.prompts["linguistic_style"].format(
            content=content,
            statistical_insights=statistical_insights
        )
        
        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            result_text = response.generations[0][0].text
            
            # Parse JSON response
            result_json = json.loads(result_text)
            
            # Create LinguisticStyle object
            linguistic_style = LinguisticStyle(
                tone=result_json.get('tone', ''),
                catchphrases=result_json.get('catchphrases', []),
                vocabulary=result_json.get('vocabulary', []),
                sentence_structures=result_json.get('sentence_structures', []),
                communication_style=result_json.get('communication_style', {})
            )
            
            self.logger.debug(f"Extracted linguistic style with {len(linguistic_style.catchphrases)} catchphrases")
            return linguistic_style
            
        except Exception as e:
            self.logger.error(f"Linguistic style extraction failed: {e}")
            # Return default linguistic style
            return LinguisticStyle(
                tone="Analysis failed - using default",
                catchphrases=[],
                vocabulary=[],
                sentence_structures=[],
                communication_style={}
            )
    
    async def _extract_mental_models(self, content: str, statistical_insights: str) -> List[MentalModel]:
        """Extract mental models using LLM"""
        
        prompt = self.prompts["mental_models"].format(
            content=content,
            statistical_insights=statistical_insights
        )
        
        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            result_text = response.generations[0][0].text
            
            # Parse JSON response
            result_json = json.loads(result_text)
            
            # Create MentalModel objects
            mental_models = []
            for model_data in result_json:
                try:
                    mental_model = MentalModel(
                        name=model_data.get('name', ''),
                        description=model_data.get('description', ''),
                        steps=model_data.get('steps', []),
                        application_contexts=model_data.get('application_contexts', []),
                        examples=model_data.get('examples', []),
                        confidence_score=model_data.get('confidence_score', 0.5)
                    )
                    mental_models.append(mental_model)
                except Exception as e:
                    self.logger.warning(f"Failed to create mental model: {e}")
                    continue
            
            # Filter by confidence threshold
            threshold = self.settings.persona_extraction.mental_models.get('min_confidence', 0.7)
            filtered_models = [m for m in mental_models if m.confidence_score >= threshold]
            
            self.logger.debug(f"Extracted {len(filtered_models)} mental models (filtered from {len(mental_models)})")
            return filtered_models
            
        except Exception as e:
            self.logger.error(f"Mental models extraction failed: {e}")
            return []
    
    async def _extract_core_beliefs(self, content: str, statistical_insights: str) -> List[CoreBelief]:
        """Extract core beliefs using LLM"""
        
        prompt = self.prompts["core_beliefs"].format(
            content=content,
            statistical_insights=statistical_insights
        )
        
        try:
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            result_text = response.generations[0][0].text
            
            # Parse JSON response
            result_json = json.loads(result_text)
            
            # Create CoreBelief objects
            core_beliefs = []
            for belief_data in result_json:
                try:
                    core_belief = CoreBelief(
                        statement=belief_data.get('statement', ''),
                        category=belief_data.get('category', ''),
                        evidence=belief_data.get('evidence', []),
                        frequency=belief_data.get('frequency', 1),
                        confidence_score=belief_data.get('confidence_score', 0.5),
                        related_mental_models=belief_data.get('related_mental_models', [])
                    )
                    core_beliefs.append(core_belief)
                except Exception as e:
                    self.logger.warning(f"Failed to create core belief: {e}")
                    continue
            
            # Filter by confidence threshold
            threshold = self.settings.persona_extraction.core_beliefs.get('min_confidence', 0.6)
            filtered_beliefs = [b for b in core_beliefs if b.confidence_score >= threshold]
            
            self.logger.debug(f"Extracted {len(filtered_beliefs)} core beliefs (filtered from {len(core_beliefs)})")
            return filtered_beliefs
            
        except Exception as e:
            self.logger.error(f"Core beliefs extraction failed: {e}")
            return []
    
    def _create_extraction_metadata(self, documents: List[Dict[str, Any]]) -> ExtractionMetadata:
        """Create extraction metadata"""
        total_time = time.time() - self.extraction_start_time if self.extraction_start_time else 0
        
        source_documents = []
        for doc in documents:
            if 'source' in doc:
                source_documents.append(doc['source'])
            elif 'file_path' in doc:
                source_documents.append(doc['file_path'])
        
        metadata = ExtractionMetadata(
            extraction_date=datetime.now(),
            extractor_version="1.0.0",
            llm_model=self.settings.llm.config.get('model', 'unknown'),
            total_processing_time=total_time,
            source_documents=source_documents,
            extraction_parameters=self.settings.persona_extraction.dict(),
            quality_scores={}  # Can be populated later with quality assessment
        )
        
        return metadata
    
    def _extract_expertise_domains(self, report: StatisticalReport) -> List[str]:
        """Extract expertise domains from statistical analysis"""
        domains = []
        
        # Extract from top keywords
        if report.top_keywords:
            business_terms = ['business', 'marketing', 'sales', 'strategy', 'growth', 'profit']
            tech_terms = ['technology', 'software', 'development', 'innovation', 'digital']
            personal_terms = ['productivity', 'habits', 'success', 'mindset', 'leadership']
            
            keywords = list(report.top_keywords.keys())
            
            if any(term in ' '.join(keywords) for term in business_terms):
                domains.append('business')
            if any(term in ' '.join(keywords) for term in tech_terms):
                domains.append('technology')
            if any(term in ' '.join(keywords) for term in personal_terms):
                domains.append('personal_development')
        
        # Extract from entities
        if report.top_entities:
            entity_text = ' '.join(report.top_entities.keys()).lower()
            
            if 'marketing' in entity_text:
                domains.append('marketing')
            if 'entrepreneur' in entity_text:
                domains.append('entrepreneurship')
            if 'invest' in entity_text:
                domains.append('investing')
        
        return list(set(domains))  # Remove duplicates
    
    def _extract_content_themes(self, report: StatisticalReport) -> List[str]:
        """Extract major content themes"""
        themes = []
        
        # Extract from collocations
        if report.top_collocations:
            for collocation in report.top_collocations[:10]:
                ngram = collocation.get('ngram', '').lower()
                
                if any(term in ngram for term in ['make money', 'build business', 'start company']):
                    themes.append('entrepreneurship')
                elif any(term in ngram for term in ['time management', 'be productive', 'get things done']):
                    themes.append('productivity') 
                elif any(term in ngram for term in ['market', 'customer', 'sell']):
                    themes.append('marketing_sales')
                elif any(term in ngram for term in ['invest', 'wealth', 'financial']):
                    themes.append('wealth_building')
        
        return list(set(themes))
    
    def _infer_communication_preferences(self, linguistic_style: LinguisticStyle) -> Dict[str, Any]:
        """Infer communication preferences from linguistic style"""
        preferences = {
            'preferred_format': 'conversational',
            'uses_storytelling': linguistic_style.communication_style.get('storytelling', 'medium') == 'high',
            'direct_communication': linguistic_style.communication_style.get('directness', 'direct') in ['direct', 'very_direct'],
            'uses_examples': linguistic_style.communication_style.get('use_of_examples', 'occasional') == 'frequent',
            'humor_level': linguistic_style.communication_style.get('humor', 'occasional'),
            'formality_level': linguistic_style.communication_style.get('formality', 'informal')
        }
        
        return preferences
    
    # Synchronous wrapper for the main extraction method
    def extract_persona_sync(self, documents: List[Dict[str, Any]]) -> PersonaConstitution:
        """Synchronous wrapper for extract_persona"""
        return asyncio.run(self.extract_persona(documents))
    
    def _estimate_processing_time(self, total_words: int, num_documents: int) -> float:
        """
        Estimate total processing time based on content metrics
        
        Args:
            total_words: Total word count across all documents
            num_documents: Number of documents
            
        Returns:
            Estimated time in minutes
        """
        # Base processing rates (conservative estimates)
        statistical_analysis_rate = 10000  # words per minute
        llm_processing_rate = 500  # words per minute for LLM analysis
        
        # Statistical analysis time
        stats_time = total_words / statistical_analysis_rate
        
        # LLM processing time (3 extraction steps)
        llm_time = (total_words / llm_processing_rate) * 3
        
        # Base overhead for model loading, etc.
        overhead_time = 2.0  # minutes
        
        # Document processing overhead
        doc_overhead = num_documents * 0.01  # 0.01 minutes per document
        
        total_time = stats_time + llm_time + overhead_time + doc_overhead
        
        return max(total_time, 1.0)  # Minimum 1 minute