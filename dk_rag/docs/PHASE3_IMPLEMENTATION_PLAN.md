# Phase 3 Implementation Plan: Agentic Architecture with Multi-Knowledge Retrieval

## Executive Summary

Phase 3 implements a sophisticated agentic system that orchestrates multi-step reasoning and tool usage to create authentic persona responses. The system uses LangChain agents to coordinate query analysis, multi-knowledge retrieval (persona data, mental models, core beliefs, and transcripts), and structured response synthesis with Chain-of-Thought reasoning.

---

## 1. Architecture Overview

### 1.1 System Components

The Phase 3 architecture consists of five core agent tools, an orchestration layer, and a synthesis engine:

```
User Query
    ↓
Query Analyzer Tool → Extracts core task & generates RAG query
    ↓
Parallel Tool Execution:
    ├── Persona Data Tool → Linguistic style & static data
    ├── Mental Models Tool → 3 most relevant frameworks  
    ├── Core Beliefs Tool → 5 most relevant principles
    └── Transcript Tool → 5 most relevant chunks
    ↓
Synthesis Engine → Chain-of-Thought reasoning → Final Response
```

### 1.2 Tool Separation Strategy

Based on modularity and debugging requirements, we implement **5 separate tools**:

1. **Query Analyzer Tool** - Preprocesses and analyzes user input
2. **Persona Data Tool** - Loads and extracts static persona information
3. **Mental Models Retriever Tool** - Retrieves relevant problem-solving frameworks
4. **Core Beliefs Retriever Tool** - Retrieves relevant foundational principles
5. **Transcript Retriever Tool** - Retrieves relevant transcript chunks

---

## 2. File and Folder Structure

### 2.1 New Directories to Create

```
dk_rag/
├── tools/                              # [NEW] Agent tools directory
│   ├── __init__.py
│   ├── base_tool.py                   # Base class for all tools
│   ├── query_analyzer_tool.py         # Query analysis and decomposition
│   ├── persona_data_tool.py           # Static persona data extraction
│   ├── mental_models_tool.py          # Mental models RAG retrieval
│   ├── core_beliefs_tool.py           # Core beliefs RAG retrieval
│   └── transcript_retriever_tool.py   # Transcript RAG retrieval
│
├── agent/                              # [NEW] Agent framework
│   ├── __init__.py
│   ├── persona_agent.py               # Main agent orchestrator
│   ├── tool_registry.py               # Tool registration and management
│   └── agent_config.py                # Agent-specific configuration
│
├── services/                           # [NEW] Service layer
│   ├── __init__.py
│   ├── query_analysis_service.py      # Query analysis logic
│   ├── knowledge_orchestrator.py      # Multi-knowledge coordination
│   └── synthesis_engine.py            # Response synthesis with CoT
│
├── api/                                # [NEW] FastAPI application
│   ├── __init__.py
│   ├── persona_api.py                 # Main API endpoints
│   ├── models.py                      # Request/response models
│   └── middleware.py                  # API middleware
│
├── prompts/agent/                      # [NEW] Agent-specific prompts
│   ├── __init__.py
│   ├── query_analysis_prompts.py      # Query decomposition prompts
│   ├── synthesis_prompts.py           # Master synthesis prompts
│   └── tool_prompts.py                # Tool-specific prompts
│
└── logging/llm_interactions/           # [NEW] LLM interaction logs
    ├── query_analysis/                 # Query analysis logs
    ├── mental_models/                  # Mental models retrieval logs
    ├── core_beliefs/                   # Core beliefs retrieval logs
    ├── transcripts/                    # Transcript retrieval logs
    └── synthesis/                      # Response synthesis logs
```

### 2.2 Modified Files

```
dk_rag/
├── config/
│   ├── settings.py                    # [MODIFY] Add agent configuration
│   └── agent_config.py                # [NEW] Agent-specific settings
│
├── models/
│   └── agent_models.py                # [NEW] Agent data models
│
└── utils/
    └── agent_utils.py                 # [NEW] Agent utility functions
```

### 2.3 Configuration Structure

```yaml
# Additional configuration in persona_config.yaml
agent:
  enabled: true
  
  query_analysis:
    llm_model: "gemini/gemini-2.0-flash"  # Fast model for analysis
    temperature: 0.3
    max_tokens: 1000
    cache_results: true
    log_interactions: true
    
  synthesis:
    llm_model: "anthropic/claude-3.5-sonnet"  # High-quality synthesis
    temperature: 0.7
    max_tokens: 4000
    use_chain_of_thought: true
    log_interactions: true
    
  tools:
    mental_models_k: 3  # Number of mental models to retrieve
    core_beliefs_k: 5   # Number of core beliefs to retrieve
    transcripts_k: 5    # Number of transcript chunks to retrieve
    parallel_execution: true
    timeout_seconds: 30
    
  logging:
    save_prompts: true
    save_responses: true
    save_extracted: true
    log_directory: "logging/llm_interactions"
```

---

## 3. Detailed Component Specifications

### 3.1 Agent Tools Implementation

#### 3.1.1 Base Tool Class
**File**: `dk_rag/tools/base_tool.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    """Base input model for all tools"""
    query: str = Field(..., description="The query or input to process")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class BasePersonaTool(BaseTool, ABC):
    """Base class for all persona agent tools"""
    
    def __init__(self, persona_id: str, settings: Settings):
        self.persona_id = persona_id
        self.settings = settings
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        super().__init__()
    
    def _run(self, query: str, metadata: Optional[Dict] = None) -> Any:
        """Execute the tool with comprehensive logging"""
        self.logger.info(f"Starting {self.name} for query: {query[:100]}...")
        
        try:
            result = self.execute(query, metadata)
            self.logger.info(f"{self.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} failed: {str(e)}", exc_info=True)
            raise
    
    @abstractmethod
    def execute(self, query: str, metadata: Optional[Dict]) -> Any:
        """Tool-specific execution logic"""
        pass
    
    def log_llm_interaction(self, prompt: str, response: str, extracted: Any):
        """Save LLM interaction for debugging"""
        # Implementation for saving to timestamped JSON files
        pass
```

#### 3.1.2 Query Analyzer Tool
**File**: `dk_rag/tools/query_analyzer_tool.py`

```python
class QueryAnalyzerTool(BasePersonaTool):
    """Analyzes user queries to extract core tasks and generate RAG queries"""
    
    name = "query_analyzer"
    description = "Extract core task and generate optimized RAG query from user input"
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze query to extract:
        - core_task: The main task the user wants to accomplish
        - rag_query: Optimized query for RAG retrieval
        - provided_context: Any context provided by the user
        - intent_type: Classification of user intent
        """
        self.logger.info("Analyzing query to extract core task")
        
        # Use LLM to analyze query
        prompt = self.build_analysis_prompt(query)
        response = self.call_llm(prompt)
        
        # Extract structured output
        extracted = self.parse_analysis_response(response)
        
        # Log the interaction
        self.log_llm_interaction(prompt, response, extracted)
        
        self.logger.info(f"Extracted core task: {extracted['core_task'][:100]}...")
        
        return extracted
```

#### 3.1.3 Persona Data Tool
**File**: `dk_rag/tools/persona_data_tool.py`

```python
class PersonaDataTool(BasePersonaTool):
    """Loads and extracts static persona data from latest artifact"""
    
    name = "persona_data"
    description = "Extract linguistic style and static persona information"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        self.artifact_discovery = ArtifactDiscovery(settings)
        self._cached_data = None
        self._cache_timestamp = None
        
    def execute(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract:
        - linguistic_style: Tone, catchphrases, vocabulary
        - communication_patterns: Speaking patterns and style
        - persona_metadata: Basic persona information
        """
        self.logger.info(f"Loading persona data for: {self.persona_id}")
        
        # Use cached data if available and fresh (within session)
        if self._cached_data and self._is_cache_valid():
            self.logger.info("Using cached persona data")
            return self._cached_data
        
        # Auto-discover and load latest artifact
        json_path, artifact_info = self.artifact_discovery.get_latest_artifact_json(self.persona_id)
        
        self.logger.info(f"Loading from artifact: {artifact_info.file_path.name}")
        
        # Extract relevant persona data
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        # Extract only linguistic style and static data
        extracted_data = {
            'linguistic_style': full_data.get('linguistic_style', {}),
            'communication_patterns': full_data.get('communication_patterns', {}),
            'persona_metadata': {
                'name': full_data.get('name'),
                'description': full_data.get('description'),
                'extraction_timestamp': artifact_info.timestamp.isoformat()
            }
        }
        
        # Cache the data
        self._cached_data = extracted_data
        self._cache_timestamp = datetime.now()
        
        # Clean up temp file if needed
        self.artifact_discovery.cleanup_temp_file(json_path)
        
        self.logger.info("Persona data extraction completed")
        
        return extracted_data
```

#### 3.1.4 Mental Models Retriever Tool
**File**: `dk_rag/tools/mental_models_tool.py`

```python
class MentalModelsRetrieverTool(BasePersonaTool):
    """Retrieves relevant mental models using RAG pipeline"""
    
    name = "mental_models_retriever"
    description = "Retrieve relevant problem-solving frameworks and methodologies"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        self.pipeline = self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize mental models RAG pipeline"""
        from ..data.storage.mental_models_store import MentalModelsStore
        from ..core.retrieval.knowledge_aware import MentalModelsPipeline
        
        store = MentalModelsStore(self.settings, self.persona_id)
        reranker = CrossEncoderReranker(self.settings)
        
        return MentalModelsPipeline(
            vector_store=store,
            reranker=reranker,
            persona_id=self.persona_id
        )
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top 3 most relevant mental models
        """
        self.logger.info("Retrieving relevant mental models")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Retrieve using pipeline
        results = self.pipeline.retrieve(
            query=rag_query,
            k=3,  # Top 3 mental models
            use_reranking=True
        )
        
        self.logger.info(f"Retrieved {len(results)} mental models")
        
        # Convert to serializable format
        return [result.to_dict() for result in results]
```

#### 3.1.5 Core Beliefs Retriever Tool
**File**: `dk_rag/tools/core_beliefs_tool.py`

```python
class CoreBeliefsRetrieverTool(BasePersonaTool):
    """Retrieves relevant core beliefs using RAG pipeline"""
    
    name = "core_beliefs_retriever"
    description = "Retrieve relevant foundational principles and beliefs"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        self.pipeline = self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize core beliefs RAG pipeline"""
        from ..data.storage.core_beliefs_store import CoreBeliefsStore
        from ..core.retrieval.knowledge_aware import CoreBeliefsPipeline
        
        store = CoreBeliefsStore(self.settings, self.persona_id)
        reranker = CrossEncoderReranker(self.settings)
        
        return CoreBeliefsPipeline(
            vector_store=store,
            reranker=reranker,
            persona_id=self.persona_id
        )
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top 5 most relevant core beliefs
        """
        self.logger.info("Retrieving relevant core beliefs")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Retrieve using pipeline
        results = self.pipeline.retrieve(
            query=rag_query,
            k=5,  # Top 5 core beliefs
            use_reranking=True
        )
        
        self.logger.info(f"Retrieved {len(results)} core beliefs")
        
        # Convert to serializable format
        return [result.to_dict() for result in results]
```

#### 3.1.6 Transcript Retriever Tool
**File**: `dk_rag/tools/transcript_retriever_tool.py`

```python
class TranscriptRetrieverTool(BasePersonaTool):
    """Retrieves relevant transcript chunks using Phase 2 advanced pipeline"""
    
    name = "transcript_retriever"
    description = "Retrieve relevant transcript chunks using advanced RAG"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        self.knowledge_indexer = KnowledgeIndexer(settings, PersonaManager(settings), persona_id)
        self.pipeline = self.knowledge_indexer.get_advanced_retrieval_pipeline(persona_id)
        
    def execute(self, query: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top 5 most relevant transcript chunks using Phase 2 pipeline
        """
        self.logger.info("Retrieving relevant transcript chunks")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Use Phase 2 advanced pipeline
        results = self.pipeline.retrieve(
            query=rag_query,
            k=5,  # Top 5 transcript chunks
            retrieval_k=25  # Candidates before reranking
        )
        
        self.logger.info(f"Retrieved {len(results)} transcript chunks")
        
        # Convert to serializable format
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': getattr(doc, 'score', None)
            }
            for doc in results
        ]
```

### 3.2 Configuration Updates

#### 3.2.1 Agent Configuration in persona_config.yaml

Add the following configuration section to `/Users/blickt/Documents/src/pdf_2_text/dk_rag/config/persona_config.yaml`:

```yaml
# Phase 3 Agent Configuration
agent:
  enabled: true
  
  # Query Analysis Tool Configuration
  query_analysis:
    llm_provider: "litellm"
    llm_model: "gemini/gemini-2.0-flash"  # Light task - fast model
    temperature: 0.3  # Lower temperature for structured extraction
    max_tokens: 1000
    timeout_seconds: 20
    max_retries: 2
    cache_results: true
    cache_ttl_hours: 24
    log_interactions: true
    
  # Synthesis Engine Configuration  
  synthesis:
    llm_provider: "litellm"
    llm_model: "gemini/gemini-2.5-pro"  # Heavy lifting - best model
    temperature: 0.7  # Higher temperature for creative synthesis
    max_tokens: 4000
    timeout_seconds: 60
    max_retries: 2
    use_chain_of_thought: true
    include_scratchpad: true
    log_interactions: true
    
  # Tool-specific Settings
  tools:
    # Persona Data Tool
    persona_data:
      cache_session: true  # Cache per session to avoid repeated loading
      cache_ttl_minutes: 60
      auto_cleanup_temp_files: true
      
    # Mental Models Retriever Tool
    mental_models:
      k: 3  # Number of mental models to retrieve
      use_reranking: true
      min_confidence_score: 0.7
      log_retrievals: true
      
    # Core Beliefs Retriever Tool
    core_beliefs:
      k: 5  # Number of core beliefs to retrieve
      use_reranking: true
      min_confidence_score: 0.6
      include_evidence: true
      log_retrievals: true
      
    # Transcript Retriever Tool
    transcripts:
      k: 5  # Number of transcript chunks to retrieve
      retrieval_k: 25  # Candidates before reranking
      use_phase2_pipeline: true  # Use full Phase 2 pipeline
      log_retrievals: true
      
    # General Tool Settings
    parallel_execution: true
    timeout_seconds: 30
    fail_fast: true  # No fallback initially - fail immediately on error
    
  # LLM Interaction Logging
  logging:
    enabled: true
    save_prompts: true
    save_responses: true
    save_extracted: true
    include_metadata: true
    log_directory: "logging/llm_interactions"  # Under base_storage_dir/persona_id/
    
    # Component-specific log directories
    components:
      query_analysis: "query_analysis"
      mental_models: "mental_models"
      core_beliefs: "core_beliefs"
      transcripts: "transcripts"
      synthesis: "synthesis"
      
  # Performance Settings
  performance:
    enable_progress_logging: true
    log_timing_metrics: true
    batch_tool_execution: false  # Sequential for now
    max_concurrent_tools: 1  # Sequential execution
    
  # Error Handling (Simplified - No Fallbacks)
  error_handling:
    use_fallbacks: false  # Disabled initially
    throw_on_error: true  # Fail fast for debugging
    include_traceback: true  # Full error details in logs
    
# FastAPI Configuration  
api:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  reload: false  # Set to true for development
  workers: 1
  
  # API Settings
  settings:
    title: "Persona Agent API"
    version: "1.0.0"
    docs_url: "/docs"
    redoc_url: "/redoc"
    
  # Rate Limiting
  rate_limiting:
    enabled: false  # Enable later for production
    requests_per_minute: 60
    
  # CORS Settings
  cors:
    enabled: true
    allow_origins: ["*"]
    allow_methods: ["GET", "POST"]
    allow_headers: ["*"]
    
  # Health Check
  health_check:
    enabled: true
    endpoint: "/health"
    include_version: true
```

#### 3.2.2 Multi-Tenant Data Path Structure

All persona-specific data follows the multi-tenant pattern:

```
{base_storage_dir}/personas/{persona_id}/
├── artifacts/              # Phase 1 JSON artifacts (auto-discovered)
├── vector_db/              # ChromaDB collections
├── indexes/                # BM25 and other indexes
│   └── bm25/              # BM25 index files
├── retrieval_cache/        # Phase 2 caches
│   ├── transcripts/
│   ├── mental_models/
│   └── core_beliefs/
├── mental_models_db/       # Mental models vector store
├── core_beliefs_db/        # Core beliefs vector store
└── logging/                # Phase 3 LLM interaction logs
    └── llm_interactions/
        ├── query_analysis/  # Query analysis logs
        ├── mental_models/   # Mental models retrieval logs
        ├── core_beliefs/    # Core beliefs retrieval logs
        ├── transcripts/     # Transcript retrieval logs
        └── synthesis/       # Response synthesis logs
```

#### 3.2.3 LiteLLM Model Strategy

All LLM calls use `litellm` as the provider with the following model assignments:

| Component | Task Type | Model | Temperature | Purpose |
|-----------|-----------|--------|-------------|---------|
| Query Analysis | Light | `gemini/gemini-2.0-flash` | 0.3 | Fast structured extraction |
| Mental Models RAG | Light | `gemini/gemini-2.0-flash` | 0.5 | Quick retrieval scoring |
| Core Beliefs RAG | Light | `gemini/gemini-2.0-flash` | 0.5 | Quick retrieval scoring |
| HyDE (Phase 2) | Light | `gemini/gemini-2.0-flash` | 0.7 | Hypothesis generation |
| Synthesis Engine | Heavy | `gemini/gemini-2.5-pro` | 0.7 | Complex reasoning & synthesis |

All models use the `GEMINI_API_KEY` environment variable for authentication.

### 3.3 Agent Orchestration Framework

#### 3.3.1 Main Persona Agent
**File**: `dk_rag/agent/persona_agent.py`

```python
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory

class PersonaAgent:
    """Main orchestrator for the persona agent system"""
    
    def __init__(self, persona_id: str, settings: Settings):
        self.persona_id = persona_id
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize synthesis engine
        self.synthesis_engine = SynthesisEngine(persona_id, settings)
        
        # Initialize LLM for agent
        self.llm = self._initialize_llm()
        
        # Create agent executor
        self.agent_executor = self._create_agent_executor()
        
        self.logger.info(f"PersonaAgent initialized for: {persona_id}")
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize all agent tools"""
        self.logger.info("Initializing agent tools")
        
        tools = [
            QueryAnalyzerTool(self.persona_id, self.settings),
            PersonaDataTool(self.persona_id, self.settings),
            MentalModelsRetrieverTool(self.persona_id, self.settings),
            CoreBeliefsRetrieverTool(self.persona_id, self.settings),
            TranscriptRetrieverTool(self.persona_id, self.settings)
        ]
        
        self.logger.info(f"Initialized {len(tools)} tools")
        return tools
    
    def process_query(self, user_query: str) -> str:
        """
        Main entry point for processing user queries
        
        Workflow:
        1. Analyze query to extract core task
        2. Retrieve persona data (parallel)
        3. Retrieve mental models (parallel)
        4. Retrieve core beliefs (parallel)
        5. Retrieve transcripts (parallel)
        6. Synthesize response with Chain-of-Thought
        """
        self.logger.info(f"Processing query: {user_query[:100]}...")
        
        try:
            # Step 1: Query Analysis
            self.logger.info("Step 1: Analyzing query")
            query_analysis = self.tools[0].execute(user_query)
            
            # Extract RAG query for retrieval
            rag_query = query_analysis.get('rag_query', user_query)
            metadata = {'rag_query': rag_query}
            
            # Step 2-5: Parallel Retrieval
            self.logger.info("Step 2-5: Executing parallel knowledge retrieval")
            
            # Execute tools in parallel (simulated - could use asyncio)
            retrieval_results = {}
            
            # Persona data
            self.logger.info("Retrieving persona data")
            retrieval_results['persona_data'] = self.tools[1].execute(rag_query, metadata)
            
            # Mental models
            self.logger.info("Retrieving mental models")
            retrieval_results['mental_models'] = self.tools[2].execute(rag_query, metadata)
            
            # Core beliefs
            self.logger.info("Retrieving core beliefs")
            retrieval_results['core_beliefs'] = self.tools[3].execute(rag_query, metadata)
            
            # Transcripts
            self.logger.info("Retrieving transcripts")
            retrieval_results['transcripts'] = self.tools[4].execute(rag_query, metadata)
            
            # Step 6: Synthesis
            self.logger.info("Step 6: Synthesizing response")
            
            response = self.synthesis_engine.synthesize(
                user_query=user_query,
                query_analysis=query_analysis,
                retrieval_results=retrieval_results
            )
            
            self.logger.info("Query processing completed successfully")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            raise
```

### 3.3 Synthesis Engine

#### 3.3.1 Response Synthesis with Chain-of-Thought
**File**: `dk_rag/services/synthesis_engine.py`

```python
class SynthesisEngine:
    """Synthesizes final responses using Chain-of-Thought reasoning"""
    
    def __init__(self, persona_id: str, settings: Settings):
        self.persona_id = persona_id
        self.settings = settings
        self.logger = get_logger(__name__)
        self.llm = self._initialize_synthesis_llm()
        
    def synthesize(
        self,
        user_query: str,
        query_analysis: Dict[str, Any],
        retrieval_results: Dict[str, Any]
    ) -> str:
        """
        Synthesize response using master prompt with Chain-of-Thought
        """
        self.logger.info("Starting response synthesis")
        
        # Build the master synthesis prompt
        prompt = self._build_synthesis_prompt(
            user_query,
            query_analysis,
            retrieval_results
        )
        
        # Call LLM for synthesis
        self.logger.info("Calling LLM for synthesis")
        response = self.llm.invoke(prompt)
        
        # Parse response to extract answer (remove scratchpad)
        final_answer = self._extract_final_answer(response)
        
        # Log the complete interaction
        self._log_synthesis_interaction(prompt, response, final_answer)
        
        self.logger.info("Synthesis completed")
        
        return final_answer
    
    def _build_synthesis_prompt(
        self,
        user_query: str,
        query_analysis: Dict,
        retrieval_results: Dict
    ) -> str:
        """Build the master synthesis prompt with constitutional rules"""
        
        template = """
You are a virtual AI persona of {persona_name}. Your goal is to respond to the user in a way that is identical to the real {persona_name} in tone, style, knowledge, and problem-solving.

### Constitutional Rules ###
- You MUST adopt the tone and style described in the <linguistic_style> context
- You MUST use appropriate catchphrases and vocabulary where natural
- You MUST NOT break character or mention that you are an AI
- You MUST apply relevant mental models from the <mental_models> context
- You MUST ensure your reasoning aligns with the <core_beliefs> context
- You MUST ground your response in facts from the <factual_context>

### Context Block ###

<linguistic_style>
{linguistic_style}
</linguistic_style>

<mental_models>
{mental_models}
</mental_models>

<core_beliefs>
{core_beliefs}
</core_beliefs>

<factual_context>
{transcripts}
</factual_context>

<user_task>
Core Task: {core_task}
Original Query: {user_query}
</user_task>

### Response Generation ###
First, think through your response step-by-step in a private scratchpad. Then write the final answer.

<scratchpad>
1. **User's Core Need**: What is the user really asking for?
2. **Relevant Mental Model**: Which framework from the context best applies?
3. **Belief Alignment**: How do the core beliefs guide my response?
4. **Factual Support**: What specific facts from transcripts support my answer?
5. **Response Structure**: How will I structure this in character?
6. **Tone Check**: Is this authentic to the persona's style?
</scratchpad>

<answer>
[Your final in-character response goes here]
</answer>
"""
        
        # Fill in the template
        filled_prompt = template.format(
            persona_name=self.persona_id,
            linguistic_style=self._format_linguistic_style(retrieval_results['persona_data']),
            mental_models=self._format_mental_models(retrieval_results['mental_models']),
            core_beliefs=self._format_core_beliefs(retrieval_results['core_beliefs']),
            transcripts=self._format_transcripts(retrieval_results['transcripts']),
            core_task=query_analysis['core_task'],
            user_query=user_query
        )
        
        return filled_prompt
```

### 3.4 FastAPI Deployment

#### 3.4.1 API Application
**File**: `dk_rag/api/persona_api.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Persona Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    persona_id: str
    metadata: Optional[Dict[str, Any]] = {}

class QueryResponse(BaseModel):
    response: str
    persona_id: str
    processing_time: float
    metadata: Dict[str, Any]

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the persona agent"""
    
    logger.info(f"API request for persona: {request.persona_id}")
    
    start_time = time.time()
    
    try:
        # Initialize agent for the persona
        agent = PersonaAgent(request.persona_id, settings)
        
        # Process the query
        response = agent.process_query(request.query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"API request completed in {processing_time:.2f}s")
        
        return QueryResponse(
            response=response,
            persona_id=request.persona_id,
            processing_time=processing_time,
            metadata={
                "query_length": len(request.query),
                "response_length": len(response)
            }
        )
        
    except Exception as e:
        logger.error(f"API request failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/v1/personas")
async def list_personas():
    """List available personas"""
    manager = PersonaManager(settings)
    personas = manager.list_personas()
    return {"personas": [p['name'] for p in personas]}
```

---

## 4. Logging Strategy

### 4.1 Progress Logging Structure

```python
# Standard progress log format for each component
self.logger.info(f"[{component_name}] Starting: {operation}")
self.logger.info(f"[{component_name}] Progress: {details}")
self.logger.info(f"[{component_name}] Completed: {result_summary}")
```

### 4.2 LLM Interaction Logging

Each LLM interaction saves three files:
```
logging/llm_interactions/{component}/{timestamp}_interaction.json
{
    "timestamp": "2025-09-25T10:30:00",
    "component": "query_analyzer",
    "persona_id": "dan_kennedy",
    "input": {
        "query": "user query text",
        "metadata": {}
    },
    "prompt": {
        "template": "prompt template used",
        "filled": "complete filled prompt sent to LLM"
    },
    "response": {
        "raw": "complete LLM response",
        "model": "gemini/gemini-2.0-flash",
        "tokens": {"input": 500, "output": 200},
        "latency_ms": 1250
    },
    "extracted": {
        "core_task": "extracted task",
        "rag_query": "optimized query"
    }
}
```

### 4.3 Error Handling

```python
try:
    # Tool operation
    result = self.execute_operation()
    self.logger.info(f"Operation succeeded: {summary}")
    
except Exception as e:
    # Log full error with traceback
    self.logger.error(
        f"Operation failed: {str(e)}",
        exc_info=True,  # Include full traceback
        extra={
            'persona_id': self.persona_id,
            'component': self.__class__.__name__,
            'operation': operation_name
        }
    )
    # Re-raise for immediate failure (no fallback)
    raise
```

---

## 5. Implementation Timeline

### Week 1: Core Tools
- Day 1-2: Base tool class and query analyzer tool
- Day 3-4: Persona data tool with artifact discovery
- Day 5: Mental models and core beliefs retriever tools

### Week 2: Agent Framework  
- Day 1-2: Transcript retriever tool
- Day 3-4: Agent orchestrator and tool registry
- Day 5: Initial integration testing

### Week 3: Synthesis Engine
- Day 1-2: Master prompt templates
- Day 3-4: Chain-of-Thought implementation
- Day 5: Response parsing and validation

### Week 4: API & Production
- Day 1-2: FastAPI endpoints
- Day 3: LLM interaction logging
- Day 4-5: End-to-end testing and optimization

---

## 6. Key Technical Decisions

### 6.1 Tool Separation
- **5 separate tools** for modularity and debugging
- Each tool can fail independently
- Clean logging boundaries
- Easier testing and maintenance

### 6.2 No Fallback/Recovery (Initially)
- Fail fast with exceptions for easier debugging
- Comprehensive error logging with full tracebacks
- Fallback mechanisms to be added after core functionality works

### 6.3 Caching Strategy
- Persona data cached per session
- Query analysis results cached
- RAG retrieval results use existing Phase 2 caching
- LLM responses logged but not cached initially

### 6.4 Logging Philosophy
- INFO level for all progress updates
- ERROR level with full traceback for failures
- Complete LLM interaction logging for debugging
- Structured JSON logs for analysis

---

## 7. Success Metrics

### Functional Requirements
- ✅ All 5 tools operational with proper error handling
- ✅ Complete LLM interaction logging
- ✅ Chain-of-Thought synthesis working
- ✅ FastAPI deployment functional

### Performance Targets
- Response time: <3 seconds (with caching)
- Tool execution: <500ms per tool
- Synthesis: <2 seconds
- Memory usage: <2GB per persona

### Quality Metrics
- Persona adherence score: >90%
- Response relevance: >85%
- Factual accuracy: 100%
- Error recovery: N/A (fail-fast initially)

---

## 8. Dependencies

### Required Libraries
```toml
# Add to pyproject.toml
langchain = "^0.3.76"
langchain-community = "^0.3.0"
fastapi = "^0.115.0"
uvicorn = "^0.32.0"
pydantic = "^2.9.0"
```

### Existing Components
- Phase 1: Persona extraction, artifacts
- Phase 2: Advanced retrieval pipeline
- Mental Models RAG: Knowledge-specific retrieval
- Core Beliefs RAG: Knowledge-specific retrieval
- Artifact Discovery: Auto-loading latest JSON

---

## 9. Risks and Mitigations

### Technical Risks
1. **LLM API failures**: Log errors, throw exception
2. **Memory issues**: Implement session-based caching
3. **Slow response times**: Add progress indicators
4. **Tool coordination issues**: Comprehensive logging

### Mitigation Strategies
- Comprehensive error logging
- Progressive implementation (tool by tool)
- Extensive local testing before deployment
- Performance monitoring at each stage

---

## 10. Next Steps

1. **Immediate**: Create directory structure
2. **Day 1**: Implement base tool class
3. **Day 2**: Implement query analyzer tool
4. **Progressive**: Add tools one by one with testing

This plan provides a clear, production-ready path to implementing Phase 3 with emphasis on modularity, comprehensive logging, and fail-fast debugging.