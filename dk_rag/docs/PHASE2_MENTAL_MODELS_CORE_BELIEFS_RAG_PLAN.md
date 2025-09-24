# Phase 2 Extension: Mental Models & Core Beliefs RAG Systems
## Production Implementation Plan

### Executive Summary

This document outlines the production-ready implementation plan for extending the Phase 2 Advanced Retrieval System to support two additional knowledge bases:
- **Mental Models RAG**: For retrieving problem-solving frameworks, methodologies, and structured approaches
- **Core Beliefs RAG**: For retrieving foundational principles, values, and belief statements with supporting evidence

The implementation maximizes reuse of existing Phase 2 components while maintaining clear separation between knowledge types and ensuring production-grade quality with comprehensive error handling, logging, and caching strategies.

---

## 1. Architecture Overview

### 1.1 System Design Principles

1. **Three Independent RAG Pipelines**: Completely separate systems with no cross-contamination
   - Transcripts RAG (existing Phase 2): HyDE → Hybrid Search → Reranking
   - Mental Models RAG (new, simplified): Vector Search → Reranking
   - Core Beliefs RAG (new, simplified): Vector Search → Reranking

2. **Selective Component Reuse**: Strategic sharing of appropriate infrastructure
   - Shared embedding model and reranker across all pipelines
   - HyDE only for transcripts (complex content)
   - BM25 only for transcripts (large document corpus)
   - Vector search for all, but simplified for mental models/core beliefs
   - Separate caching per knowledge type

3. **Production Requirements**:
   - Comprehensive error handling with graceful degradation
   - Structured logging with knowledge-type context
   - Performance monitoring and metrics collection
   - Cache isolation between knowledge types
   - Atomic operations with rollback capability
   - Resource cleanup and memory management

### 1.2 Data Flow Architecture

**Transcripts Pipeline (Full Phase 2)**:
```
Transcript Documents → Vector Store + BM25 Store
        ↓
Query → HyDE → Hybrid Search → Reranking → Results
```

**Mental Models Pipeline (Simplified)**:
```
Phase 1 JSON → Mental Models Processor → Document Builder → Vector Store
        ↓
Query → Vector Search → Reranking → Results
```

**Core Beliefs Pipeline (Simplified)**:
```
Phase 1 JSON → Core Beliefs Processor → Document Builder → Vector Store
        ↓
Query → Vector Search → Reranking → Results
```

### 1.3 Cache Separation Strategy

**Cache Directory Structure**:
```
/retrieval_cache/
├── transcripts/
│   ├── hyde_cache/        # HyDE hypothesis generation cache
│   ├── rerank_cache/      # Cross-encoder reranking cache
│   ├── bm25_cache/        # BM25 search cache
│   └── pipeline_logs/     # Full pipeline execution logs
├── mental_models/
│   ├── rerank_cache/      # Only reranking cache (no HyDE or BM25)
│   └── pipeline_logs/     # Simplified pipeline logs
└── core_beliefs/
    ├── rerank_cache/      # Only reranking cache (no HyDE or BM25)
    └── pipeline_logs/     # Simplified pipeline logs
```

**Cache Key Format**:
```
{knowledge_type}:{persona_id}:{operation}:{content_hash}
```

---

## 2. Detailed Implementation Components

### 2.1 Data Layer Components

#### 2.1.1 Knowledge Processors

**Purpose**: Extract and transform Phase 1 artifacts into indexable documents

**Key Features**:
- Robust JSON parsing with schema validation
- Graceful handling of missing or malformed data
- Preservation of all metadata for reconstruction
- Batch processing with progress tracking
- Memory-efficient streaming for large datasets

**Error Handling**:
- Schema validation before processing
- Field-level error recovery
- Detailed error reporting with context
- Automatic retry with exponential backoff
- Fallback to partial data extraction

#### 2.1.2 Document Builders

**Purpose**: Create optimized searchable documents for each knowledge type

**Mental Models Document Structure**:
```python
{
    "content": "{name}\n{description}\nSteps:\n{formatted_steps}\nCategories: {categories}",
    "metadata": {
        "type": "mental_model",
        "name": str,
        "description": str,
        "steps": List[str],
        "categories": List[str],
        "confidence_score": float,
        "frequency": int,
        "persona_id": str,
        "source_file": str,
        "extraction_timestamp": str
    }
}
```

**Core Beliefs Document Structure**:
```python
{
    "content": "{statement}\nCategory: {category}\nEvidence:\n{formatted_evidence}",
    "metadata": {
        "type": "core_belief",
        "statement": str,
        "category": str,
        "supporting_evidence": List[str],
        "confidence_score": float,
        "frequency": int,
        "persona_id": str,
        "source_file": str,
        "extraction_timestamp": str
    }
}
```

### 2.2 Storage Layer Components

#### 2.2.1 Multi-Knowledge Vector Store

**Features**:
- Separate ChromaDB collections per knowledge type
- Atomic collection creation and deletion
- Transaction support for bulk operations
- Collection versioning for rollback capability
- Automatic backup before modifications

**Collection Naming Convention**:
```
{persona_id}_{knowledge_type}_v{version}
```

**Error Handling**:
- Collection existence validation
- Duplicate prevention with idempotent operations
- Corrupted index recovery
- Automatic collection repair
- Disk space monitoring

#### 2.2.2 Multi-BM25 Store

**Features**:
- Independent BM25 indexes per knowledge type
- Compressed index storage
- Incremental index updates
- Index integrity verification
- Hot-swappable index loading

**Index Path Structure**:
```
/indexes/
├── {persona_id}/
│   ├── transcripts/
│   │   ├── bm25_index.pkl.gz
│   │   └── bm25_metadata.json
│   ├── mental_models/
│   │   ├── bm25_index.pkl.gz
│   │   └── bm25_metadata.json
│   └── core_beliefs/
│       ├── bm25_index.pkl.gz
│       └── bm25_metadata.json
```

### 2.3 Retrieval Layer Components

#### 2.3.1 Knowledge-Type Aware Retrievers

**Mental Models Retriever**:
- Prioritizes exact framework name matches
- Weights results by frequency of use
- Groups related models by category
- Returns complete model structure
- Handles partial query matches

**Core Beliefs Retriever**:
- Prioritizes high-confidence beliefs
- Weights by supporting evidence strength
- Groups beliefs by category
- Semantic similarity for belief alignment
- Context-aware belief selection

#### 2.3.2 Independent Pipeline Implementations

**Mental Models Pipeline**:
- Direct vector similarity search using query embedding
- Cross-encoder reranking for precision
- Returns complete framework structure with steps and categories
- No HyDE (frameworks are already well-structured)
- No BM25 (semantic similarity sufficient for small, curated set)

**Core Beliefs Pipeline**:
- Direct vector similarity search using query embedding
- Cross-encoder reranking weighted by confidence scores
- Returns belief statements with supporting evidence
- No HyDE (beliefs are already clear statements)
- No BM25 (semantic similarity sufficient for principle matching)

**Pipeline Isolation**:
```python
# Each pipeline is completely independent
class MentalModelsPipeline:
    def __init__(self, vector_store, reranker, cache):
        # Only vector store and reranker - no HyDE or BM25
        
class CoreBeliefsPipeline:
    def __init__(self, vector_store, reranker, cache):
        # Only vector store and reranker - no HyDE or BM25
        
class TranscriptsPipeline(AdvancedRetrievalPipeline):
    # Full Phase 2 pipeline unchanged
```

### 2.4 Caching Strategy

#### 2.4.1 Cache Separation Implementation

**Cache Manager Updates**:
```python
class MultiKnowledgeRetrievalCache:
    def __init__(self, base_cache_dir: str):
        self.caches = {
            KnowledgeType.TRANSCRIPTS: RetrievalCache(f"{base_cache_dir}/transcripts"),
            KnowledgeType.MENTAL_MODELS: RetrievalCache(f"{base_cache_dir}/mental_models"),
            KnowledgeType.CORE_BELIEFS: RetrievalCache(f"{base_cache_dir}/core_beliefs")
        }
    
    def get_cache(self, knowledge_type: KnowledgeType) -> RetrievalCache:
        """Returns appropriate cache for knowledge type"""
        
    def cache_hyde_generation(self, knowledge_type: KnowledgeType, ...):
        """Cache HyDE results with knowledge type namespace"""
        
    def cache_reranking(self, knowledge_type: KnowledgeType, ...):
        """Cache reranking results with knowledge type namespace"""
```

**Cache Key Generation**:
```python
def generate_cache_key(
    knowledge_type: KnowledgeType,
    persona_id: str,
    operation: str,
    content: str
) -> str:
    """
    Generates unique cache key including knowledge type.
    
    Format: {knowledge_type}:{persona_id}:{operation}:{content_hash}
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{knowledge_type.value}:{persona_id}:{operation}:{content_hash}"
```

#### 2.4.2 Cache Lifecycle Management

**Features**:
- TTL-based expiration per knowledge type
- LRU eviction with configurable limits
- Cache warming for frequently accessed items
- Cache statistics per knowledge type
- Automated cleanup scheduling

### 2.5 Logging Strategy

#### 2.5.1 Structured Logging

**Log Entry Format**:
```python
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "component": "MultiKnowledgePipeline",
    "knowledge_type": "mental_models",
    "persona_id": "dan_kennedy",
    "operation": "retrieve",
    "query": "productivity framework",
    "duration_ms": 245,
    "results_count": 5,
    "cache_hit": true,
    "metadata": {
        "hyde_used": true,
        "reranking_used": true,
        "fallback_triggered": false
    }
}
```

#### 2.5.2 Log Aggregation

**Log Directory Structure**:
```
/logs/
├── application/
│   └── dk_rag_{date}.log
├── retrieval/
│   ├── transcripts/
│   ├── mental_models/
│   └── core_beliefs/
└── performance/
    └── metrics_{date}.json
```

### 2.6 Error Handling Strategy

#### 2.6.1 Error Hierarchy

```python
class DKRAGError(Exception):
    """Base exception for all DK RAG errors"""

class KnowledgeIndexingError(DKRAGError):
    """Errors during knowledge base indexing"""

class RetrievalError(DKRAGError):
    """Errors during retrieval operations"""

class KnowledgeTypeError(DKRAGError):
    """Invalid knowledge type operations"""

class CacheError(DKRAGError):
    """Cache-related errors"""
```

#### 2.6.2 Graceful Degradation

**Fallback Chain**:
1. Try advanced pipeline with all features
2. Fallback to hybrid search without reranking
3. Fallback to vector-only search
4. Fallback to BM25-only search
5. Return empty results with error context

**Error Recovery**:
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientError)
)
def retrieve_with_retry(self, query: str, knowledge_type: KnowledgeType):
    """Retrieval with automatic retry for transient errors"""
```

---

## 3. API and Interface Design

### 3.1 Knowledge Indexer API Extensions

```python
class KnowledgeIndexer:
    # New methods for multi-knowledge support
    
    def build_mental_models_index(
        self,
        persona_id: str,
        json_path: str,
        rebuild: bool = False,
        validate: bool = True
    ) -> IndexingResult:
        """
        Build mental models knowledge base from Phase 1 JSON.
        
        Args:
            persona_id: Unique persona identifier
            json_path: Path to Phase 1 JSON artifact
            rebuild: Force rebuild existing index
            validate: Validate JSON schema before processing
            
        Returns:
            IndexingResult with statistics and any errors
        """
    
    def build_core_beliefs_index(
        self,
        persona_id: str,
        json_path: str,
        rebuild: bool = False,
        validate: bool = True
    ) -> IndexingResult:
        """
        Build core beliefs knowledge base from Phase 1 JSON.
        """
    
    def search_mental_models(
        self,
        query: str,
        persona_id: str,
        k: int = 5,
        use_hyde: bool = True,
        use_reranking: bool = True,
        return_metadata: bool = True
    ) -> List[MentalModelResult]:
        """
        Search mental models with full pipeline.
        """
    
    def search_core_beliefs(
        self,
        query: str,
        persona_id: str,
        k: int = 5,
        use_hyde: bool = True,
        use_reranking: bool = True,
        return_metadata: bool = True
    ) -> List[CoreBeliefResult]:
        """
        Search core beliefs with full pipeline.
        """
    
    def unified_search(
        self,
        query: str,
        persona_id: str,
        knowledge_types: List[KnowledgeType] = None,
        k_per_type: int = 3,
        aggregate_results: bool = True
    ) -> UnifiedSearchResult:
        """
        Search across multiple knowledge types.
        """
```

### 3.2 CLI Commands

```bash
# Building knowledge bases
dk-rag build-mental-models \
    --persona-id dan_kennedy \
    --json-path /path/to/persona.json \
    --rebuild \
    --validate

dk-rag build-core-beliefs \
    --persona-id dan_kennedy \
    --json-path /path/to/persona.json \
    --rebuild \
    --validate

# Searching knowledge bases
dk-rag search \
    --knowledge-type mental_models \
    --persona-id dan_kennedy \
    --query "productivity framework" \
    --top-k 5 \
    --use-advanced

# Unified search across all knowledge types
dk-rag unified-search \
    --persona-id dan_kennedy \
    --query "how to increase sales" \
    --include transcripts,mental_models,core_beliefs \
    --k-per-type 3

# Management commands
dk-rag list-indexes --persona-id dan_kennedy
dk-rag validate-index --knowledge-type mental_models --persona-id dan_kennedy
dk-rag clear-cache --knowledge-type all --persona-id dan_kennedy
dk-rag export-index --knowledge-type core_beliefs --persona-id dan_kennedy --output /path/to/export
```

### 3.3 Configuration Schema

```yaml
# retrieval_config.yaml
retrieval:
  # Existing Phase 2 config...
  
  # Multi-knowledge configuration
  knowledge_types:
    transcripts:
      enabled: true
      collection_suffix: "_transcripts"
      cache_ttl_hours: 168
      max_cache_size_mb: 500
      
    mental_models:
      enabled: true
      collection_suffix: "_mental_models"
      cache_ttl_hours: 336  # 2 weeks
      max_cache_size_mb: 200
      weight_by_frequency: true
      include_steps: true
      min_confidence_score: 0.7
      
    core_beliefs:
      enabled: true
      collection_suffix: "_core_beliefs"
      cache_ttl_hours: 336  # 2 weeks
      max_cache_size_mb: 200
      weight_by_confidence: true
      include_evidence: true
      min_confidence_score: 0.8
      evidence_limit: 5
  
  # Query routing configuration
  query_routing:
    auto_detect: true
    default_types: ["transcripts"]
    keyword_patterns:
      mental_models:
        - "framework"
        - "method"
        - "approach"
        - "steps"
        - "process"
        - "system"
      core_beliefs:
        - "believe"
        - "principle"
        - "value"
        - "philosophy"
        - "mindset"
        - "conviction"
  
  # Unified search configuration
  unified_search:
    parallel_execution: true
    result_fusion: "weighted"
    type_weights:
      transcripts: 0.5
      mental_models: 0.3
      core_beliefs: 0.2
    deduplication_threshold: 0.85
```

---

## 4. Complete File Structure Changes

### 4.1 New Files to Create (22 files)

```
dk_rag/
├── core/
│   ├── knowledge_builders/           # NEW DIRECTORY
│   │   ├── __init__.py               # NEW: Module initialization
│   │   ├── base_builder.py           # NEW: Abstract base class for builders
│   │   ├── mental_models_builder.py  # NEW: Mental models document builder
│   │   └── core_beliefs_builder.py   # NEW: Core beliefs document builder
│   │
│   └── retrieval/
│       ├── knowledge_aware/          # NEW DIRECTORY
│       │   ├── __init__.py           # NEW: Module initialization
│       │   ├── mental_models_retriever.py  # NEW: Mental models retriever
│       │   ├── core_beliefs_retriever.py   # NEW: Core beliefs retriever
│       │   ├── multi_knowledge_pipeline.py # NEW: Unified pipeline
│       │   └── query_router.py       # NEW: Query intent detection & routing
│       │
│       └── cache/                    # NEW DIRECTORY
│           ├── __init__.py           # NEW: Module initialization
│           └── multi_knowledge_cache.py    # NEW: Knowledge-aware caching
│
├── data/
│   ├── processing/
│   │   ├── persona_knowledge_processor.py  # NEW: Process Phase 1 artifacts
│   │   └── knowledge_validator.py    # NEW: Validate knowledge data
│   │
│   └── storage/
│       ├── multi_knowledge_store.py  # NEW: Multi-collection vector store
│       └── multi_bm25_store.py       # NEW: Multi-index BM25 store
│
├── models/
│   ├── knowledge_types.py            # NEW: Enums and type definitions
│   └── knowledge_results.py          # NEW: Result data models
│
├── cli/
│   └── knowledge_builder.py          # NEW: CLI for knowledge operations
│
└── tests/
    └── retrieval/
        ├── test_multi_knowledge.py   # NEW: Integration tests
        ├── test_mental_models.py     # NEW: Mental models tests
        └── test_core_beliefs.py      # NEW: Core beliefs tests
```

### 4.2 Modified Files (8 files)

```
dk_rag/
├── core/
│   ├── knowledge_indexer.py          # MODIFY: Add multi-knowledge methods
│   └── retrieval/
│       ├── __init__.py               # MODIFY: Export new components
│       ├── advanced_pipeline.py      # MODIFY: Add knowledge_type parameter
│       └── hyde_retriever.py         # MODIFY: Add knowledge-aware prompts
│
├── config/
│   └── retrieval_config.py           # MODIFY: Add knowledge configs
│
├── data/
│   └── storage/
│       └── retrieval_cache.py        # MODIFY: Add knowledge namespacing
│
├── prompts/
│   └── hyde_prompts.py               # MODIFY: Add knowledge-specific prompts
│
└── utils/
    └── logging.py                    # MODIFY: Add knowledge context
```

### 4.3 Directory Structure Summary

**New Directories (4)**:
- `dk_rag/core/knowledge_builders/`
- `dk_rag/core/retrieval/knowledge_aware/`
- `dk_rag/core/retrieval/cache/`

**Total Changes**:
- **22 new files**
- **8 modified files**
- **4 new directories**

---

## 5. Implementation Phases

### Phase 5.1: Foundation (Week 1)
1. Create base data models and enums (`knowledge_types.py`, `knowledge_results.py`)
2. Implement knowledge processors (`persona_knowledge_processor.py`)
3. Build document builders (`mental_models_builder.py`, `core_beliefs_builder.py`)
4. Set up multi-storage infrastructure (`multi_knowledge_store.py`, `multi_bm25_store.py`)

### Phase 5.2: Retrieval Layer (Week 2)
1. Implement specialized retrievers (mental models and core beliefs)
2. Build query router with intent detection
3. Create unified multi-knowledge pipeline
4. Integrate knowledge-aware caching

### Phase 5.3: Integration (Week 3)
1. Update KnowledgeIndexer with new methods
2. Modify existing components for knowledge awareness
3. Update configuration system
4. Implement CLI commands

### Phase 5.4: Testing & Optimization (Week 4)
1. Comprehensive unit tests
2. Integration testing
3. Performance benchmarking
4. Cache optimization
5. Error recovery testing
6. Production deployment preparation

---

## 6. Testing Strategy

### 6.1 Unit Tests
- Document builder validation
- Storage operations (CRUD)
- Retrieval accuracy
- Cache operations
- Error handling

### 6.2 Integration Tests
- End-to-end pipeline testing
- Multi-knowledge unified search
- Cache coherence across knowledge types
- Fallback mechanism validation
- Concurrent access testing

### 6.3 Performance Tests
- Indexing speed benchmarks
- Query latency measurements
- Memory usage profiling
- Cache hit rate analysis
- Scalability testing (10K+ documents)

### 6.4 Production Validation
- Data integrity checks
- Rollback procedures
- Resource cleanup
- Memory leak detection
- Load testing

---

## 7. Monitoring & Operations

### 7.1 Metrics Collection
```python
# Metrics to track per knowledge type
metrics = {
    "indexing": {
        "documents_processed": int,
        "indexing_duration_ms": float,
        "errors_encountered": int,
        "index_size_mb": float
    },
    "retrieval": {
        "queries_processed": int,
        "avg_latency_ms": float,
        "cache_hit_rate": float,
        "fallback_rate": float
    },
    "cache": {
        "total_entries": int,
        "size_mb": float,
        "eviction_count": int,
        "hit_rate": float
    }
}
```

### 7.2 Health Checks
```python
def health_check(knowledge_type: KnowledgeType) -> HealthStatus:
    """
    Comprehensive health check for knowledge base.
    
    Checks:
    - Index availability
    - Vector store connectivity
    - BM25 index integrity
    - Cache accessibility
    - Recent query performance
    """
```

### 7.3 Alerting Thresholds
- Query latency > 1000ms
- Cache hit rate < 60%
- Index corruption detected
- Disk usage > 90%
- Memory usage > 80%
- Error rate > 5%

---

## 8. Security Considerations

### 8.1 Data Protection
- Encryption at rest for sensitive beliefs/models
- Access control per persona
- Audit logging for data access
- PII detection and masking

### 8.2 Input Validation
- Query sanitization
- JSON schema validation
- Path traversal prevention
- Injection attack prevention

### 8.3 Resource Limits
- Query timeout enforcement
- Result size limits
- Cache size caps
- Concurrent request throttling

---

## 9. Performance Optimization

### 9.1 Indexing Optimization
- Batch document processing
- Parallel index building
- Incremental updates
- Background index optimization

### 9.2 Retrieval Optimization
- Query result caching
- Embedding precomputation
- Index warming
- Connection pooling

### 9.3 Memory Management
- Lazy loading of indexes
- LRU cache eviction
- Memory-mapped file usage
- Garbage collection tuning

---

## 10. Migration & Rollback

### 10.1 Migration Strategy
1. Backup existing indexes
2. Create new knowledge indexes
3. Validate new indexes
4. Update routing configuration
5. Enable new knowledge types
6. Monitor for issues
7. Remove backup after validation

### 10.2 Rollback Procedure
1. Disable new knowledge types
2. Restore routing to transcripts-only
3. Clear new indexes if needed
4. Restore from backup if corruption
5. Restart services
6. Validate functionality

---

## 11. Documentation Requirements

### 11.1 Code Documentation
- Comprehensive docstrings for all public methods
- Type hints for all parameters and returns
- Usage examples in docstrings
- Error conditions documented

### 11.2 API Documentation
- OpenAPI/Swagger spec for REST endpoints
- CLI command documentation
- Configuration reference
- Migration guides

### 11.3 Operational Documentation
- Deployment procedures
- Monitoring setup
- Troubleshooting guides
- Performance tuning guide

---

## 12. Success Criteria

### 12.1 Functional Requirements
- ✅ Successfully index mental models from Phase 1 JSON
- ✅ Successfully index core beliefs from Phase 1 JSON
- ✅ Retrieve relevant mental models with >80% precision
- ✅ Retrieve relevant core beliefs with >80% precision
- ✅ Unified search across all knowledge types
- ✅ Complete structure preservation in results

### 12.2 Performance Requirements
- ✅ Indexing: <10ms per document
- ✅ Query latency: <500ms p95
- ✅ Cache hit rate: >70%
- ✅ Memory usage: <2GB per persona
- ✅ Concurrent queries: 100+ QPS

### 12.3 Reliability Requirements
- ✅ 99.9% availability
- ✅ Graceful degradation on component failure
- ✅ Data durability (no data loss)
- ✅ Automatic error recovery
- ✅ Complete audit trail

---

## Appendix A: Example Usage

### A.1 Building Mental Models Index
```python
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.config.settings import Settings

# Initialize indexer
settings = Settings()
indexer = KnowledgeIndexer(settings, persona_manager)

# Build mental models index
result = indexer.build_mental_models_index(
    persona_id="dan_kennedy",
    json_path="/path/to/persona_dan_kennedy.json",
    rebuild=False,
    validate=True
)

print(f"Indexed {result.documents_count} mental models")
print(f"Errors: {result.errors}")
```

### A.2 Searching Core Beliefs
```python
# Search core beliefs
beliefs = indexer.search_core_beliefs(
    query="What beliefs about marketing success?",
    persona_id="dan_kennedy",
    k=5,
    use_hyde=True,
    use_reranking=True
)

for belief in beliefs:
    print(f"Statement: {belief.statement}")
    print(f"Confidence: {belief.confidence_score}")
    print(f"Evidence: {belief.supporting_evidence}")
```

### A.3 Unified Multi-Knowledge Search
```python
# Search across all knowledge types
results = indexer.unified_search(
    query="How to build a successful business?",
    persona_id="dan_kennedy",
    knowledge_types=["transcripts", "mental_models", "core_beliefs"],
    k_per_type=3,
    aggregate_results=True
)

print(f"Found {len(results.transcripts)} transcript matches")
print(f"Found {len(results.mental_models)} relevant frameworks")
print(f"Found {len(results.core_beliefs)} supporting beliefs")
```

---

## Appendix B: Configuration Examples

### B.1 Production Configuration
```yaml
retrieval:
  knowledge_types:
    mental_models:
      enabled: true
      cache_ttl_hours: 336
      max_cache_size_mb: 500
      min_confidence_score: 0.8
      batch_size: 100
      parallel_workers: 4
      error_retry_attempts: 3
      error_retry_delay: 2
```

### B.2 Development Configuration
```yaml
retrieval:
  knowledge_types:
    mental_models:
      enabled: true
      cache_ttl_hours: 1
      max_cache_size_mb: 50
      min_confidence_score: 0.5
      batch_size: 10
      parallel_workers: 1
      error_retry_attempts: 1
      error_retry_delay: 0
```

---

This comprehensive plan provides a production-ready blueprint for implementing the Mental Models and Core Beliefs RAG systems, with proper separation, error handling, logging, and all the infrastructure needed for a robust deployment.