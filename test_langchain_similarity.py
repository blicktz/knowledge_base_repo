#!/usr/bin/env python3
"""
Temporary script to test LangChain Chroma similarity methods
and verify which methods are available for getting scores.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dk_rag.config.settings import Settings
from dk_rag.data.storage.langchain_vector_store import LangChainVectorStore

def test_langchain_similarity_methods():
    """Test various LangChain similarity methods to see which ones work."""
    
    # Initialize settings and vector store
    print("=" * 80)
    print("Testing LangChain Chroma Similarity Methods")
    print("=" * 80)
    
    try:
        # Load settings from file like the working script does
        config_file = "dk_rag/config/persona_config.yaml"
        settings = Settings.from_file(config_file)
        vector_store = LangChainVectorStore(settings, persona_id="greg_startup")
        
        query = "how do you build an audience"
        k = 5
        
        print(f"\nQuery: '{query}'")
        print(f"K: {k}")
        print("-" * 40)
        
        # Get the actual LangChain Chroma vector store
        chroma_store = vector_store.vector_store
        
        # Check if we need the correct collection
        if chroma_store._collection.count() == 0:
            print("Current collection is empty, trying greg_startup_documents...")
            try:
                from langchain_chroma import Chroma
                
                # Try to connect to the collection that actually has data
                data_store = Chroma(
                    collection_name="greg_startup_documents",
                    embedding_function=vector_store.vector_store.embeddings,
                    persist_directory=str(Path("/Volumes/J15/aicallgo_data/persona_data_base/personas/greg_startup/vector_db"))
                )
                
                print(f"Data collection count: {data_store._collection.count()}")
                if data_store._collection.count() > 0:
                    print("Found data collection! Using that for tests...")
                    chroma_store = data_store
                    
            except Exception as e:
                print(f"Could not access data collection: {e}")
                print("Continuing with original collection...")
        
        print(f"Vector store type: {type(chroma_store)}")
        
        # Check collection info
        print(f"Collection name: {chroma_store._collection.name}")
        print(f"Document count: {chroma_store._collection.count()}")
        
        print(f"Available methods:")
        methods = [method for method in dir(chroma_store) if 'similarity' in method.lower()]
        for method in methods:
            print(f"  - {method}")
        
        print("\n" + "="*60)
        print("TEST 1: similarity_search (basic)")
        print("="*60)
        
        try:
            docs = chroma_store.similarity_search(query, k=k)
            print(f"✓ similarity_search: Found {len(docs)} documents")
            for i, doc in enumerate(docs[:2]):
                print(f"  [{i+1}] {doc.page_content[:100]}...")
        except Exception as e:
            print(f"✗ similarity_search failed: {e}")
        
        print("\n" + "="*60)
        print("TEST 2: similarity_search_with_score")
        print("="*60)
        
        try:
            results = chroma_store.similarity_search_with_score(query, k=k)
            print(f"✓ similarity_search_with_score: Found {len(results)} results")
            for i, (doc, score) in enumerate(results[:3]):
                print(f"  [{i+1}] Score: {score:.4f} | {doc.page_content[:80]}...")
        except Exception as e:
            print(f"✗ similarity_search_with_score failed: {e}")
        
        print("\n" + "="*60)
        print("TEST 3: similarity_search_with_relevance_scores")
        print("="*60)
        
        try:
            results = chroma_store.similarity_search_with_relevance_scores(query, k=k)
            print(f"✓ similarity_search_with_relevance_scores: Found {len(results)} results")
            for i, (doc, score) in enumerate(results[:3]):
                print(f"  [{i+1}] Relevance: {score:.4f} | {doc.page_content[:80]}...")
        except Exception as e:
            print(f"✗ similarity_search_with_relevance_scores failed: {e}")
        
        print("\n" + "="*60)
        print("TEST 4: Check for vector-based scoring methods")
        print("="*60)
        
        # Test if any vector-based scoring methods exist
        vector_methods = [method for method in dir(chroma_store) if 'vector' in method.lower() and 'score' in method.lower()]
        if vector_methods:
            print(f"Found vector-based scoring methods: {vector_methods}")
            
            # Try to get an embedding to test with
            try:
                embedding = chroma_store.embeddings.embed_query(query)
                print(f"Generated embedding of length: {len(embedding)}")
                
                for method_name in vector_methods:
                    try:
                        method = getattr(chroma_store, method_name)
                        results = method(embedding, k=k)
                        print(f"✓ {method_name}: Found {len(results)} results")
                        if results and hasattr(results[0], '__len__') and len(results[0]) == 2:
                            print(f"  First score: {results[0][1]}")
                    except Exception as e:
                        print(f"✗ {method_name} failed: {e}")
                        
            except Exception as e:
                print(f"Could not test vector methods (embedding failed): {e}")
        else:
            print("No vector-based scoring methods found")
        
        print("\n" + "="*60)
        print("TEST 5: Test max_marginal_relevance_search")
        print("="*60)
        
        try:
            results = chroma_store.max_marginal_relevance_search(query, k=k)
            print(f"✓ max_marginal_relevance_search: Found {len(results)} results")
            for i, doc in enumerate(results[:2]):
                print(f"  [{i+1}] {doc.page_content[:80]}...")
        except Exception as e:
            print(f"✗ max_marginal_relevance_search failed: {e}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("This test helps us determine:")
        print("1. Which similarity methods are actually available")
        print("2. Which methods return scores vs just documents")
        print("3. What the score ranges and formats are")
        print("4. Whether we need custom implementation or can use built-in methods")
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_langchain_similarity_methods()
    sys.exit(0 if success else 1)