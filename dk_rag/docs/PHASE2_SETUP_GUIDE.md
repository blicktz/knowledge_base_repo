# Phase 2 Advanced Retrieval System - End-to-End Setup Guide

## Overview

This guide provides step-by-step instructions for setting up and testing the Phase 2 advanced retrieval system. Phase 2 implements a sophisticated 3-stage pipeline that achieves **60-80% improvement** over basic vector search through:

1. **HyDE (Hypothetical Document Embeddings)** - LLM-generated hypothetical answers for better semantic search
2. **Hybrid Search (BM25 + Vector)** - Combines keyword-based BM25 with semantic vector search
3. **Cross-Encoder Reranking** - Precision refinement using specialized reranking models

## Implementation Summary

Based on the `PHASE2_IMPLEMENTATION.md`, we have implemented:

- **HyDE Retriever** with comprehensive LLM logging and caching
- **BM25 Store** using bm25s library (500x faster than rank-bm25)
- **Hybrid Retriever** with weighted score fusion and RRF support
- **Cross-Encoder Reranker** with local and API backends
- **Advanced Pipeline** orchestrating all components with error handling
- **Retrieval Cache** with multi-level caching and performance monitoring

## Prerequisites & Dependencies

### Required Libraries

All necessary libraries are already installed in your project:

```toml
# Phase 2 Core Dependencies (from pyproject.toml)
bm25s = "^0.2.14"           # Fast BM25 implementation
rerankers = "^0.10.0"       # Cross-encoder reranking models
cohere = "^5.18.0"          # Cohere API for reranking
langchain = "^0.3.0"        # LLM orchestration
chromadb = "^0.6.1"         # Vector database
sentence-transformers = "^5.1.0"  # Embedding models
```

### Verify Installation

```bash
# Verify Phase 2 packages are installed
poetry run python -c "import bm25s; import rerankers; import cohere; print('✅ All Phase 2 packages available')"
```

### Required Models

These models will be downloaded automatically on first use:

```python
# Embedding Model (for vector search)
"sentence-transformers/all-mpnet-base-v2"

# Reranking Model (for cross-encoder reranking)
"mixedbread-ai/mxbai-rerank-large-v1"

# SpaCy Model (for text processing)
# Download with: python -m spacy download en_core_web_sm
"en_core_web_sm"
```

## Environment Setup

### 1. API Keys Configuration

Create a `.env` file in your project root:

```bash
# OpenRouter API (for HyDE hypothesis generation)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Cohere API (optional, for reranking API backend)
COHERE_API_KEY=your_cohere_api_key_here

# OpenAI API (fallback option)
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Storage Directory Setup

The configuration uses persistent storage at:
```bash
/Volumes/J15/aicallgo_data/persona_data_base
```

Create the required directory structure:
```bash
mkdir -p /Volumes/J15/aicallgo_data/persona_data_base/indexes/bm25
mkdir -p /Volumes/J15/aicallgo_data/persona_data_base/retrieval_cache
mkdir -p /Volumes/J15/aicallgo_data/persona_data_base/personas
```

### 3. Verify Configuration

The Phase 2 configuration is already set in `dk_rag/config/persona_config.yaml`:

```yaml
retrieval:
  enabled: true  # Phase 2 is enabled
  
  # HyDE Configuration
  hyde:
    enabled: true
    prompt_template: "default"
    cache_size: 128
    auto_select_prompt: true
    
  # Hybrid Search Configuration
  hybrid_search:
    enabled: true
    bm25_weight: 0.4      # BM25 influence
    vector_weight: 0.6    # Vector influence
    
  # Reranking Configuration
  reranking:
    enabled: true
    model: "mixedbread-ai/mxbai-rerank-large-v1"
    device: "auto"        # Auto-detect GPU/CPU
    top_k: 5
```

## Step-by-Step Setup

### Step 1: Verify Dependencies

```python
# Create a test script: test_phase2_setup.py
from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.core.knowledge_indexer import KnowledgeIndexer

def verify_phase2_setup():
    try:
        # Load settings
        settings = Settings.from_config_file("dk_rag/config/persona_config.yaml")
        print("✅ Configuration loaded successfully")
        
        # Check Phase 2 imports
        from dk_rag.data.storage.bm25_store import BM25Store
        from dk_rag.core.retrieval.hyde_retriever import HyDERetriever
        from dk_rag.core.retrieval.hybrid_retriever import HybridRetriever
        from dk_rag.core.retrieval.reranker import CrossEncoderReranker
        from dk_rag.core.retrieval.advanced_pipeline import AdvancedRetrievalPipeline
        print("✅ All Phase 2 components imported successfully")
        
        # Check external libraries
        import bm25s
        import rerankers
        print("✅ External libraries available")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_phase2_setup()
```

Run the verification:
```bash
poetry run python test_phase2_setup.py
```

### Step 2: Initialize Persona Manager

```python
# Create initialization script: setup_phase2_test.py
from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.core.knowledge_indexer import KnowledgeIndexer

def initialize_test_environment():
    # Load configuration
    settings = Settings.from_config_file("dk_rag/config/persona_config.yaml")
    
    # Initialize persona manager
    persona_manager = PersonaManager(settings)
    
    # Create test persona
    test_persona_id = persona_manager.register_persona("Test Phase 2 Persona")
    print(f"✅ Created test persona: {test_persona_id}")
    
    # Initialize knowledge indexer with Phase 2
    indexer = KnowledgeIndexer(settings, persona_manager, test_persona_id)
    print("✅ Knowledge indexer initialized with Phase 2 support")
    
    return indexer, test_persona_id

if __name__ == "__main__":
    indexer, persona_id = initialize_test_environment()
```

### Step 3: Prepare Test Data

```python
# Add to setup_phase2_test.py
def prepare_test_documents(indexer, persona_id):
    # Sample test documents for Phase 2 testing
    test_documents = [
        {
            "content": "Productivity frameworks are systematic approaches to organizing work and maximizing efficiency. The Getting Things Done (GTD) methodology focuses on capturing all tasks in external systems, while the Eisenhower Matrix helps prioritize tasks based on urgency and importance. Time blocking involves dedicating specific time slots to different activities.",
            "metadata": {"source": "productivity_guide", "category": "frameworks"}
        },
        {
            "content": "Effective time management techniques include the Pomodoro Technique, which uses 25-minute focused work sessions followed by short breaks. Deep work practices require sustained concentration and elimination of distractions. Calendar blocking helps protect time for important tasks and reduces context switching.",
            "metadata": {"source": "time_management", "category": "techniques"}
        },
        {
            "content": "Goal setting frameworks like SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound) provide structure for achievement. OKRs (Objectives and Key Results) align team efforts toward common outcomes. Regular review cycles ensure goals remain relevant and progress is tracked.",
            "metadata": {"source": "goal_setting", "category": "frameworks"}
        },
        {
            "content": "Learning optimization strategies include spaced repetition for long-term retention, active recall for strengthening memory, and interleaving different topics to improve understanding. The Feynman Technique involves explaining concepts in simple terms to identify knowledge gaps.",
            "metadata": {"source": "learning_methods", "category": "strategies"}
        },
        {
            "content": "Decision-making frameworks help reduce cognitive load and improve outcomes. The DECIDE model (Define, Explore, Consider, Identify, Develop, Evaluate) provides a systematic approach. Mental models like first principles thinking break down complex problems into fundamental truths.",
            "metadata": {"source": "decision_making", "category": "frameworks"}
        }
    ]
    
    # Add documents to the knowledge base
    chunk_ids = indexer.vector_store.add_documents(test_documents)
    print(f"✅ Added {len(chunk_ids)} test documents to vector store")
    
    # Build BM25 index for hybrid search
    doc_texts = [doc["content"] for doc in test_documents]
    doc_ids = [f"doc_{i}" for i in range(len(test_documents))]
    
    # Initialize BM25 store
    bm25_store = indexer.get_bm25_store()
    bm25_store.build_index(doc_texts, doc_ids)
    print("✅ Built BM25 index for hybrid search")
    
    return test_documents
```

### Step 4: Test Individual Components

```python
# Add component testing to setup_phase2_test.py
def test_individual_components(indexer):
    """Test each Phase 2 component individually"""
    
    # Test 1: HyDE Retriever
    print("\n🧪 Testing HyDE Retriever...")
    hyde_retriever = indexer.get_hyde_retriever()
    hypothesis = hyde_retriever.generate_hypothesis("What are the best productivity techniques?")
    print(f"✅ HyDE generated hypothesis: {hypothesis[:100]}...")
    
    # Test 2: BM25 Search
    print("\n🧪 Testing BM25 Search...")
    bm25_store = indexer.get_bm25_store()
    bm25_results = bm25_store.search("productivity frameworks", k=3)
    print(f"✅ BM25 found {len(bm25_results)} results")
    
    # Test 3: Vector Search
    print("\n🧪 Testing Vector Search...")
    vector_results = indexer.vector_store.search("time management techniques", n_results=3)
    print(f"✅ Vector search found {len(vector_results)} results")
    
    # Test 4: Hybrid Search
    print("\n🧪 Testing Hybrid Search...")
    hybrid_retriever = indexer.get_hybrid_retriever()
    hybrid_results = hybrid_retriever.search("goal setting methods", k=3)
    print(f"✅ Hybrid search found {len(hybrid_results)} results")
    
    # Test 5: Reranking
    print("\n🧪 Testing Cross-Encoder Reranking...")
    reranker = indexer.get_reranker()
    reranked_results = reranker.rerank("decision making frameworks", hybrid_results, top_k=2)
    print(f"✅ Reranker refined to {len(reranked_results)} top results")
    
    return True
```

### Step 5: End-to-End Pipeline Testing

```python
def test_end_to_end_pipeline(indexer):
    """Test the complete Phase 2 pipeline"""
    
    print("\n🚀 Testing Complete Phase 2 Pipeline...")
    
    # Get advanced retrieval pipeline
    pipeline = indexer.get_advanced_retrieval_pipeline()
    
    # Test queries that demonstrate different aspects
    test_queries = [
        "What are the most effective productivity frameworks?",
        "How can I improve my time management skills?",
        "What goal setting techniques work best?",
        "How do I make better decisions?",
        "What learning strategies are most effective?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test Query {i}: {query}")
        
        # Run complete pipeline
        start_time = time.time()
        results = pipeline.retrieve(query, k=3, retrieval_k=10)
        end_time = time.time()
        
        print(f"⏱️  Pipeline execution time: {end_time - start_time:.3f}s")
        print(f"📊 Retrieved {len(results)} final results")
        
        # Show top result
        if results:
            top_result = results[0]
            print(f"🥇 Top result: {top_result.page_content[:100]}...")
            
            # Show metadata
            metadata = top_result.metadata
            print(f"📝 Metadata: {dict(list(metadata.items())[:3])}...")
    
    return True

# Complete test function
def run_complete_phase2_test():
    """Run complete Phase 2 setup and testing"""
    import time
    
    print("🚀 Starting Phase 2 Complete Setup and Testing")
    print("=" * 60)
    
    # Step 1: Initialize
    indexer, persona_id = initialize_test_environment()
    
    # Step 2: Prepare data
    test_docs = prepare_test_documents(indexer, persona_id)
    
    # Step 3: Test components
    test_individual_components(indexer)
    
    # Step 4: End-to-end testing
    test_end_to_end_pipeline(indexer)
    
    print("\n✅ Phase 2 Complete Setup and Testing Finished!")
    print("🎯 Your Phase 2 advanced retrieval system is ready!")

if __name__ == "__main__":
    run_complete_phase2_test()
```

## Running the End-to-End Test

1. **Save the setup script** as `setup_phase2_test.py` in your project root
2. **Run the complete test**:
   ```bash
   poetry run python setup_phase2_test.py
   ```

Expected output:
```
🚀 Starting Phase 2 Complete Setup and Testing
============================================================
✅ Created test persona: test_phase_2_persona
✅ Knowledge indexer initialized with Phase 2 support
✅ Added 5 test documents to vector store
✅ Built BM25 index for hybrid search

🧪 Testing HyDE Retriever...
✅ HyDE generated hypothesis: Productivity frameworks are systematic methodologies...

🧪 Testing BM25 Search...
✅ BM25 found 3 results

🧪 Testing Vector Search...
✅ Vector search found 3 results

🧪 Testing Hybrid Search...
✅ Hybrid search found 3 results

🧪 Testing Cross-Encoder Reranking...
✅ Reranker refined to 2 top results

🚀 Testing Complete Phase 2 Pipeline...
📋 Test Query 1: What are the most effective productivity frameworks?
⏱️  Pipeline execution time: 0.456s
📊 Retrieved 3 final results
🥇 Top result: Productivity frameworks are systematic approaches to organizing work...
```

## Performance Comparison

To see the improvement over Phase 1, you can compare results:

```python
def compare_phase1_vs_phase2(indexer, query="What are effective productivity techniques?"):
    """Compare Phase 1 vs Phase 2 results"""
    
    # Phase 1: Basic vector search
    phase1_results = indexer.vector_store.search(query, n_results=5)
    
    # Phase 2: Advanced pipeline
    pipeline = indexer.get_advanced_retrieval_pipeline()
    phase2_results = pipeline.retrieve(query, k=5)
    
    print(f"Phase 1 Results: {len(phase1_results)} documents")
    print(f"Phase 2 Results: {len(phase2_results)} documents")
    
    # Show quality comparison
    print("\n📊 Quality Comparison:")
    print("Phase 1 - Top result:", phase1_results[0]['document'][:100] + "...")
    print("Phase 2 - Top result:", phase2_results[0].page_content[:100] + "...")
```

## Configuration Options

### For Different Use Cases

**High Precision (Quality over Speed):**
```yaml
retrieval:
  hyde:
    enabled: true
    auto_select_prompt: true
  hybrid_search:
    bm25_weight: 0.3
    vector_weight: 0.7
  reranking:
    enabled: true
    top_k: 3
```

**Balanced Performance:**
```yaml
retrieval:
  hyde:
    enabled: true
  hybrid_search:
    bm25_weight: 0.4
    vector_weight: 0.6
  reranking:
    enabled: true
    top_k: 5
```

**High Speed (Speed over Quality):**
```yaml
retrieval:
  hyde:
    enabled: false  # Skip HyDE for speed
  hybrid_search:
    enabled: true
  reranking:
    enabled: false  # Skip reranking for speed
```

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'bm25s'**
```bash
poetry add bm25s@^0.2.14
```

**2. CUDA out of memory**
```yaml
reranking:
  device: "cpu"  # Force CPU usage
  batch_size: 16  # Reduce batch size
```

**3. API Key errors**
```bash
# Check your .env file has:
OPENROUTER_API_KEY=your_key_here
```

**4. Permission denied for storage**
```bash
sudo mkdir -p /Volumes/J15/aicallgo_data/persona_data_base
sudo chown -R $USER /Volumes/J15/aicallgo_data/persona_data_base
```

### Debugging Commands

```python
# Check Phase 2 status
from dk_rag.config.retrieval_config import Phase2RetrievalConfig
config = Phase2RetrievalConfig.from_env()
print(f"Phase 2 enabled: {config.enabled}")

# Check component initialization
indexer = KnowledgeIndexer(settings, persona_manager)
stats = indexer.get_phase2_statistics()
print(f"Components initialized: {stats}")

# Check cache performance
cache_dir = "/Volumes/J15/aicallgo_data/persona_data_base/retrieval_cache"
# Look for pipeline logs and cache statistics
```

## Production Deployment

For production use, consider:

1. **Persistent Storage**: Use reliable storage with backups
2. **API Rate Limits**: Configure retries and fallbacks
3. **Monitoring**: Set up logging and alerting
4. **Scaling**: Consider async processing for high volume
5. **Caching**: Tune cache sizes for your workload

## Next Steps

After successful setup, you can:

1. **Add Real Data**: Replace test documents with your actual content
2. **Tune Configuration**: Adjust weights and parameters for your use case
3. **Monitor Performance**: Track improvements and bottlenecks
4. **Scale Up**: Add more advanced features as needed

## Expected Results

With Phase 2 properly configured, you should see:

- **60-80% improvement** in result relevance
- **Better semantic understanding** through HyDE
- **Enhanced keyword matching** via BM25
- **Precision refinement** through reranking
- **Comprehensive logging** for analysis and debugging

Your Phase 2 advanced retrieval system is now ready for production use!

---

## ⚠️ **IMPORTANT: Multi-Tenant Storage Issue Identified**

### Current Problem: Phase 2 is NOT Multi-Tenant

After code analysis, **Phase 2 components currently store data in global/shared locations** rather than persona-specific directories like Phase 1 does.

#### Current Multi-Tenant Structure (Phase 1 ✅)
Phase 1 properly implements multi-tenancy through `PersonaManager`:

```
/Volumes/J15/aicallgo_data/persona_data_base/
├── persona_registry.json
├── persona_1/
│   ├── vector_db/          # ✅ Persona-specific ChromaDB
│   └── artifacts/          # ✅ Persona-specific extractions
├── persona_2/
│   ├── vector_db/          # ✅ Isolated from persona_1
│   └── artifacts/          # ✅ Isolated from persona_1
└── ...
```

#### Current Phase 2 Problem (❌ Global Storage)
Phase 2 components currently use **shared paths**:

```
/Volumes/J15/aicallgo_data/persona_data_base/
├── indexes/bm25/           # ❌ SHARED across all personas
├── retrieval_cache/        # ❌ SHARED across all personas
└── ...
```

**This means:**
- **BM25 indexes are shared** - Persona A can search Persona B's documents
- **Retrieval cache is shared** - Cross-persona data leakage
- **Pipeline logs are shared** - No isolation
- **Violates multi-tenant architecture**

---

## 🔧 **Required Fixes Implementation Plan**

### Fix #1: Make Phase 2 Multi-Tenant (Persona-Aware)

⚠️ **BREAKING CHANGE**: The `get_bm25_index_path()` and `get_cache_dir()` methods now require `persona_id` parameter. Any direct calls to these methods without `persona_id` will throw a `ValueError`. This enforces proper multi-tenant usage and prevents data mixing between personas.

#### 1.1 Update `StorageConfig` Class
```python
# In dk_rag/config/retrieval_config.py
def get_bm25_index_path(self, persona_id: Optional[str] = None) -> Path:
    """Get persona-specific BM25 index path"""
    if self.bm25_index_path:
        return Path(self.bm25_index_path)
    
    if not persona_id:
        raise ValueError("persona_id is required - single-tenant mode is no longer supported")
    
    base_dir = Path(self.base_storage_dir)
    return base_dir / "personas" / persona_id / "indexes" / "bm25"

def get_cache_dir(self, persona_id: Optional[str] = None) -> Path:
    """Get persona-specific cache directory"""
    if self.cache_dir:
        return Path(self.cache_dir)
    
    if not persona_id:
        raise ValueError("persona_id is required - single-tenant mode is no longer supported")
    
    base_dir = Path(self.base_storage_dir)
    return base_dir / "personas" / persona_id / "retrieval_cache"
```

#### 1.2 Update `KnowledgeIndexer.setup_phase2_retrieval()`
```python
# In dk_rag/core/knowledge_indexer.py
def setup_phase2_retrieval(self):
    """Initialize Phase 2 advanced retrieval components with persona awareness."""
    try:
        self.logger.info("Setting up Phase 2 advanced retrieval...")
        
        # Load retrieval configuration
        if hasattr(self.settings, 'retrieval'):
            self.retrieval_config = self.settings.retrieval
        else:
            self.retrieval_config = Phase2RetrievalConfig.from_env()
        
        # Setup persona-specific storage paths
        base_storage = self.retrieval_config.storage.base_storage_dir
        bm25_path = self.retrieval_config.storage.get_bm25_index_path(self.persona_id)
        cache_dir = self.retrieval_config.storage.get_cache_dir(self.persona_id)
        
        # Initialize persona-specific cache
        if self.retrieval_config.caching.enabled:
            self.retrieval_cache = RetrievalCache(
                str(cache_dir),
                cache_size=self.retrieval_config.caching.hyde_cache_size,
                ttl_hours=self.retrieval_config.caching.cache_ttl_hours,
                enable_compression=self.retrieval_config.caching.enable_compression
            )
            self.logger.debug(f"Retrieval cache initialized for persona: {self.persona_id}")
        
        # Initialize persona-specific BM25 store
        if self.retrieval_config.hybrid_search.enabled:
            self.bm25_store = BM25Store(
                str(bm25_path),
                k1=self.retrieval_config.hybrid_search.bm25_k1,
                b=self.retrieval_config.hybrid_search.bm25_b
            )
            self.logger.debug(f"BM25 store initialized for persona: {self.persona_id}")
        
        self.logger.info(f"Phase 2 retrieval setup complete for persona: {self.persona_id}")
        
    except Exception as e:
        self.logger.error(f"Failed to setup Phase 2 retrieval for persona {self.persona_id}: {e}")
        if self.retrieval_config:
            self.retrieval_config.enabled = False
```

#### 1.3 Expected Directory Structure After Fix
```
/Volumes/J15/aicallgo_data/persona_data_base/
├── persona_registry.json
├── persona_1/
│   ├── vector_db/              # Phase 1 (existing)
│   ├── artifacts/              # Phase 1 (existing)
│   ├── indexes/bm25/           # Phase 2 (new) ✅ ISOLATED
│   └── retrieval_cache/        # Phase 2 (new) ✅ ISOLATED
│       ├── hyde/
│       ├── rerank/
│       └── pipeline_logs/
├── persona_2/
│   ├── vector_db/              # Phase 1 (existing)
│   ├── artifacts/              # Phase 1 (existing)
│   ├── indexes/bm25/           # Phase 2 (new) ✅ ISOLATED
│   └── retrieval_cache/        # Phase 2 (new) ✅ ISOLATED
└── ...
```

### Fix #2: Auto-Build BM25 Indexes in `make build-kb`

#### 2.1 Integrate into `build_knowledge_base()` Method
```python
# In dk_rag/core/knowledge_indexer.py
def build_knowledge_base(self, 
                       documents_dir: str,
                       persona_id: Optional[str] = None,
                       rebuild: bool = False,
                       file_pattern: str = "*.md") -> Dict[str, Any]:
    # ... existing vector store building code ...
    
    # NEW: Build Phase 2 indexes automatically
    phase2_results = {}
    if self.retrieval_config and self.retrieval_config.enabled:
        self.logger.info("Building Phase 2 indexes...")
        try:
            phase2_results = self.build_phase2_indexes(persona_id, rebuild)
            self.logger.info("✅ Phase 2 indexes built successfully")
        except Exception as e:
            self.logger.error(f"Phase 2 index building failed: {e}")
            phase2_results = {"error": str(e)}
    
    # Update results to include Phase 2 stats
    results.update({
        'phase2_enabled': self.retrieval_config.enabled if self.retrieval_config else False,
        'phase2_results': phase2_results
    })
    
    return results
```

#### 2.2 Enhanced `build_phase2_indexes()` with Progress Indicators
```python
def build_phase2_indexes(self, persona_id: Optional[str] = None, rebuild: bool = False):
    """Build Phase 2 indexes (BM25) with progress indicators for large document sets."""
    if not self.retrieval_config or not self.retrieval_config.enabled:
        self.logger.warning("Phase 2 not enabled, skipping index building")
        return {}
    
    persona_id = persona_id or self.persona_id
    if not persona_id:
        raise ValueError("persona_id required for Phase 2 index building")
    
    # Get persona-specific vector store
    vector_store = self.persona_manager.get_persona_vector_store(persona_id)
    
    # Get all documents from vector store for BM25 indexing
    self.logger.info("Retrieving documents for BM25 index building...")
    all_chunks = vector_store.get_all_documents()  # New method needed
    
    if not all_chunks:
        self.logger.warning("No documents found in vector store for BM25 indexing")
        return {"bm25_documents": 0}
    
    # Build BM25 index with progress indicators
    if self.retrieval_config.hybrid_search.enabled and self.bm25_store:
        self.logger.info(f"Building BM25 index for {len(all_chunks)} documents...")
        
        # Extract text and IDs with progress
        doc_texts = []
        doc_ids = []
        
        from tqdm import tqdm
        for i, chunk in enumerate(tqdm(all_chunks, desc="Processing documents for BM25")):
            doc_texts.append(chunk.get('document', chunk.get('content', '')))
            doc_ids.append(chunk.get('id', f"doc_{i}"))
        
        # Build the index
        self.logger.info("Building BM25 search index...")
        self.bm25_store.build_index(doc_texts, doc_ids, rebuild=rebuild)
        
        bm25_stats = self.bm25_store.get_statistics()
        self.logger.info(f"✅ BM25 index built: {bm25_stats['num_documents']} documents")
        
        return {
            "bm25_documents": bm25_stats['num_documents'],
            "bm25_statistics": bm25_stats
        }
    
    return {"bm25_enabled": False}
```

#### 2.3 Update CLI Output in `persona_builder.py`
```python
def build_knowledge_base(self, args):
    """Build or rebuild the knowledge base"""
    # ... existing code ...
    
    results = self.knowledge_indexer.build_knowledge_base(
        documents_dir=args.documents_dir,
        rebuild=args.rebuild,
        file_pattern=args.pattern
    )
    
    # Display results
    print("\nKnowledge Base Built Successfully!")
    print("-" * 40)
    print(f"Documents loaded: {results['documents_loaded']}")
    print(f"Total words: {results['total_words']:,}")
    print(f"Chunks created: {results['chunks_created']}")
    print(f"Total chunks in database: {results['collection_stats'].get('total_chunks', 0)}")
    
    # NEW: Display Phase 2 results
    if results.get('phase2_enabled', False):
        print(f"\nPhase 2 Advanced Retrieval:")
        print("-" * 30)
        phase2_results = results.get('phase2_results', {})
        if 'error' in phase2_results:
            print(f"❌ Phase 2 setup failed: {phase2_results['error']}")
        else:
            print(f"✅ BM25 index built: {phase2_results.get('bm25_documents', 0)} documents")
            print("✅ Hybrid search ready")
            print("✅ Advanced pipeline initialized")
    
    if args.verbose:
        # ... existing verbose output ...
        if results.get('phase2_enabled', False):
            print("\nPhase 2 Statistics:")
            bm25_stats = results.get('phase2_results', {}).get('bm25_statistics', {})
            for key, value in bm25_stats.items():
                print(f"  {key}: {value}")
```

---

## 🚀 **Implementation Status**

### Current Status: ⚠️ **NEEDS IMPLEMENTATION**

The issues identified above need to be implemented before Phase 2 can be safely used in a multi-tenant environment.

### After Implementation: ✅ **Expected Results**

1. **Full Persona Isolation**: Each persona's Phase 2 data completely isolated
2. **Automatic BM25 Building**: `make build-kb` automatically builds BM25 indexes  
3. **Progress Indicators**: Clear progress for large document sets (1000+ docs)
4. **Enforced Multi-tenancy**: Configuration methods now require persona_id (breaking change for direct API usage)
5. **Enhanced CLI Output**: Shows Phase 2 status and statistics

### Quick Test After Implementation
```bash
# Test multi-tenant isolation
make build-kb PERSONA_NAME=persona_1 DOCS_DIR=/path/to/docs1
make build-kb PERSONA_NAME=persona_2 DOCS_DIR=/path/to/docs2

# Verify isolation - should show different document counts
ls -la /Volumes/J15/aicallgo_data/persona_data_base/personas/persona_1/indexes/bm25/
ls -la /Volumes/J15/aicallgo_data/persona_data_base/personas/persona_2/indexes/bm25/
```

---

**⚠️ Until these fixes are implemented, Phase 2 should not be used in multi-tenant environments as it will cause data leakage between personas.**