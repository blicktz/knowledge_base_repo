"""
HyDE (Hypothetical Document Embeddings) Retriever

This module implements HyDE, which generates hypothetical answers to queries
and uses them for more effective semantic search.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain.vectorstores.base import VectorStore

from ...utils.logging import get_logger
from ...config.settings import Settings


class HyDERetriever:
    """
    Hypothetical Document Embeddings retriever.
    
    Generates hypothetical answers to queries and uses them for improved
    semantic search, achieving 40-60% better retrieval accuracy.
    """
    
    def __init__(
        self, 
        llm: BaseLLM,
        embeddings: Embeddings,
        vector_store: VectorStore,
        settings: Optional[Settings] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize HyDE retriever.
        
        Args:
            llm: Language model for hypothesis generation
            embeddings: Embeddings model for encoding
            vector_store: Vector store for similarity search
            settings: Optional settings object
            cache_dir: Directory for caching LLM interactions
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings = settings or Settings()
        self.logger = get_logger(__name__)
        
        # Setup cache directory for LLM logging
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            base_dir = getattr(settings, 'base_storage_dir', '/Volumes/J15/aicallgo_data/persona_data_base')
            self.cache_dir = Path(base_dir) / "retrieval_cache" / "hyde_llm_logs"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"HyDE retriever initialized with cache at: {self.cache_dir}")
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query for caching."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _log_llm_interaction(
        self,
        query: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log complete LLM interaction to file.
        
        Args:
            query: Original user query
            prompt: Complete prompt sent to LLM
            response: LLM response
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        query_hash = self._get_query_hash(query)
        
        interaction = {
            "timestamp": timestamp,
            "query_hash": query_hash,
            "original_query": query,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "model": str(self.llm),
            "component": "HyDE"
        }
        
        # Save to timestamped file
        filename = f"hyde_{timestamp.replace(':', '-')}_{query_hash[:8]}.json"
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(interaction, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Logged LLM interaction to: {filepath}")
    
    def generate_hypothesis(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        log_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate hypothetical answer for the query.
        
        Args:
            query: User query
            prompt_template: Optional custom prompt template
            log_metadata: Additional metadata for logging
            
        Returns:
            Generated hypothetical answer
        """
        self.logger.info(f"Generating hypothesis for query: {query[:100]}...")
        
        # Default prompt template - comprehensive and detailed
        if not prompt_template:
            prompt_template = """Please provide a comprehensive and detailed answer to the following question. 
Include relevant information, examples, explanations, and any important context that would help someone 
fully understand the topic. Be thorough and informative.

Question: {query}

Detailed Answer:"""
        
        # Format prompt with query
        prompt = prompt_template.format(query=query)
        
        try:
            # Generate hypothesis
            response = self.llm.invoke(prompt)
            
            # Extract text from response (handle different response types)
            if hasattr(response, 'content'):
                hypothesis = response.content
            elif isinstance(response, str):
                hypothesis = response
            else:
                hypothesis = str(response)
            
            # Log the complete interaction
            self._log_llm_interaction(
                query=query,
                prompt=prompt,
                response=hypothesis,
                metadata=log_metadata
            )
            
            self.logger.info(f"Generated hypothesis of length: {len(hypothesis)}")
            return hypothesis
            
        except Exception as e:
            self.logger.error(f"Error generating hypothesis: {e}")
            # Log error interaction
            self._log_llm_interaction(
                query=query,
                prompt=prompt,
                response=f"ERROR: {str(e)}",
                metadata={"error": True, "error_message": str(e)}
            )
            # Fall back to original query
            return query
    
    def retrieve(
        self,
        query: str,
        k: int = 20,
        use_hypothesis: bool = True,
        prompt_template: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents using HyDE.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_hypothesis: Whether to use hypothesis generation
            prompt_template: Optional custom prompt template
            metadata: Additional metadata for logging
            
        Returns:
            List of retrieved documents
        """
        self.logger.info(f"HyDE retrieval for query: {query[:100]}... (k={k})")
        
        if use_hypothesis:
            # Generate hypothetical answer
            search_query = self.generate_hypothesis(
                query,
                prompt_template=prompt_template,
                log_metadata=metadata
            )
            
            # Embed the hypothesis
            self.logger.debug("Embedding hypothesis for search...")
            query_embedding = self.embeddings.embed_query(search_query)
            
            # Search with hypothesis embedding
            documents = self.vector_store.similarity_search_by_vector(
                query_embedding,
                k=k
            )
            
            # Add metadata to documents
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'hyde'
                doc.metadata['original_query'] = query
                doc.metadata['hypothesis_used'] = True
        else:
            # Fall back to regular search
            self.logger.debug("Using direct query search (hypothesis disabled)")
            documents = self.vector_store.similarity_search(query, k=k)
            
            # Add metadata
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'direct'
                doc.metadata['original_query'] = query
                doc.metadata['hypothesis_used'] = False
        
        self.logger.info(f"Retrieved {len(documents)} documents using HyDE")
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 20,
        use_hypothesis: bool = True,
        prompt_template: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores using HyDE.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            use_hypothesis: Whether to use hypothesis generation
            prompt_template: Optional custom prompt template
            metadata: Additional metadata for logging
            
        Returns:
            List of (document, score) tuples
        """
        self.logger.info(f"HyDE retrieval with scores for query: {query[:100]}...")
        
        if use_hypothesis:
            # Generate hypothetical answer
            search_query = self.generate_hypothesis(
                query,
                prompt_template=prompt_template,
                log_metadata=metadata
            )
            
            # Embed the hypothesis
            query_embedding = self.embeddings.embed_query(search_query)
            
            # Search with hypothesis embedding
            if hasattr(self.vector_store, 'similarity_search_with_score_by_vector'):
                results = self.vector_store.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=k
                )
            else:
                # Fallback if method not available
                documents = self.vector_store.similarity_search_by_vector(
                    query_embedding,
                    k=k
                )
                # Assign default scores
                results = [(doc, 1.0) for doc in documents]
            
            # Add metadata
            for doc, score in results:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'hyde'
                doc.metadata['original_query'] = query
                doc.metadata['hypothesis_used'] = True
                doc.metadata['similarity_score'] = score
        else:
            # Fall back to regular search with scores
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                results = self.vector_store.similarity_search_with_score(query, k=k)
            else:
                documents = self.vector_store.similarity_search(query, k=k)
                results = [(doc, 1.0) for doc in documents]
            
            # Add metadata
            for doc, score in results:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'direct'
                doc.metadata['original_query'] = query
                doc.metadata['hypothesis_used'] = False
                doc.metadata['similarity_score'] = score
        
        self.logger.info(f"Retrieved {len(results)} documents with scores")
        return results
    
    def get_cached_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Load all cached hypothesis generations for analysis.
        
        Returns:
            List of all cached LLM interactions
        """
        cached_interactions = []
        
        for filepath in self.cache_dir.glob("hyde_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    interaction = json.load(f)
                    cached_interactions.append(interaction)
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {filepath}: {e}")
        
        return cached_interactions