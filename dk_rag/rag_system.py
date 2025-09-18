"""
Simple RAG system using scikit-learn for text similarity
"""

import os
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def format_excerpt_preview(content: str, source_info: str, max_chars: int = 300) -> str:
    """Format excerpt preview showing beginning and end"""
    if len(content) <= max_chars * 2:
        # Short excerpt - show full content
        return f"FULL CONTENT: \"{content.strip()}\""
    
    beginning = content[:max_chars].strip()
    ending = content[-max_chars:].strip()
    
    return f"""BEGINNING: "{beginning}..."
[...MIDDLE CONTENT OMITTED...]
ENDING: "...{ending}\""""


class SimpleRAG:
    """
    A simple Retrieval-Augmented Generation system using TF-IDF and cosine similarity
    """
    
    def __init__(self, config_path: str):
        """Initialize the RAG system with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.documents = []
        self.doc_vectors = None
        self.is_indexed = False
        self.db_path = Path(__file__).parent / "db"
        self.db_path.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        for key, value in config.get('llm', {}).get('config', {}).items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config['llm']['config'][key] = os.getenv(env_var, '')
                
        for key, value in config.get('embedder', {}).get('config', {}).items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config['embedder']['config'][key] = os.getenv(env_var, '')
        
        return config
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunk_size = self.config.get('rag', {}).get('chunk_size', 1000)
        chunk_overlap = self.config.get('rag', {}).get('chunk_overlap', 200)
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            
        return chunks
    
    def add_documents(self, documents_dir: str):
        """Add documents from a directory to the RAG system"""
        print(f"Loading documents from {documents_dir}...")
        
        documents_path = Path(documents_dir)
        if not documents_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
        
        # Load all .md files
        for file_path in documents_path.glob("*.md"):
            print(f"Processing {file_path.name}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = self._chunk_text(content)
                
                # Add chunks as separate documents
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 100:  # Only add substantial chunks
                        doc_metadata = {
                            'source': file_path.name,
                            'chunk_id': i,
                            'content': chunk
                        }
                        self.documents.append(doc_metadata)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Loaded {len(self.documents)} chunks from {len(list(documents_path.glob('*.md')))} documents")
        
        # Create vectors
        self._create_index()
        
    def _create_index(self):
        """Create TF-IDF index of documents"""
        if not self.documents:
            print("No documents to index")
            return
            
        print("Creating document index...")
        
        # Extract text content for vectorization
        texts = [doc['content'] for doc in self.documents]
        
        # Fit vectorizer and transform documents
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.is_indexed = True
        
        # Save index to disk
        self._save_index()
        
        print(f"Index created with {len(texts)} documents")
    
    def _save_index(self):
        """Save the index to disk"""
        index_data = {
            'documents': self.documents,
            'vectorizer': self.vectorizer,
            'doc_vectors': self.doc_vectors
        }
        
        with open(self.db_path / "index.pkl", 'wb') as f:
            pickle.dump(index_data, f)
    
    def _load_index(self):
        """Load the index from disk"""
        index_path = self.db_path / "index.pkl"
        if not index_path.exists():
            return False
            
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.vectorizer = index_data['vectorizer']
            self.doc_vectors = index_data['doc_vectors']
            self.is_indexed = True
            
            print(f"Loaded existing index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def query(self, query_text: str, max_results: int = None, debug: bool = False) -> List[str]:
        """Query the RAG system for relevant documents"""
        if not self.is_indexed:
            if not self._load_index():
                return []
        
        max_results = max_results or self.config.get('rag', {}).get('max_results', 5)
        similarity_threshold = self.config.get('rag', {}).get('similarity_threshold', 0.1)
        
        # Transform query
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Debug information
        if debug:
            max_sim = np.max(similarities)
            avg_sim = np.mean(similarities)
            print(f"DEBUG: Query: '{query_text}'")
            print(f"DEBUG: Max similarity: {max_sim:.4f}")
            print(f"DEBUG: Average similarity: {avg_sim:.4f}")
            print(f"DEBUG: Threshold: {similarity_threshold}")
            print(f"DEBUG: Documents above threshold: {np.sum(similarities >= similarity_threshold)}")
        
        # Get top results above threshold
        top_indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in top_indices[:max_results * 2]:  # Check more candidates
            if similarities[idx] >= similarity_threshold:
                content = self.documents[idx]['content']
                results.append(content)
                if debug:
                    source_info = f"{self.documents[idx]['source']}_chunk_{self.documents[idx]['chunk_id']}"
                    print(f"DEBUG: Added result with similarity {similarities[idx]:.4f}")
                    print(f"SOURCE: {source_info}")
                    print("PREVIEW:")
                    print(format_excerpt_preview(content, source_info))
                    print("---")
        
        # If no results and similarity threshold is too high, lower it temporarily
        if not results and similarity_threshold > 0.05:
            if debug:
                print("DEBUG: No results found, trying with lower threshold...")
            for idx in top_indices[:max_results]:
                if similarities[idx] >= 0.05:  # Much lower fallback threshold
                    content = self.documents[idx]['content']
                    results.append(content)
                    if debug:
                        source_info = f"{self.documents[idx]['source']}_chunk_{self.documents[idx]['chunk_id']}"
                        print(f"DEBUG: Fallback result with similarity {similarities[idx]:.4f}")
                        print(f"SOURCE: {source_info}")
                        print("PREVIEW:")
                        print(format_excerpt_preview(content, source_info))
                        print("---")
        
        return results[:max_results]
    
    def get_context(self, query_text: str, debug: bool = False) -> str:
        """Get formatted context for the query"""
        relevant_chunks = self.query(query_text, debug=debug)
        
        if not relevant_chunks:
            if debug:
                print("DEBUG: No relevant chunks found")
            return "No relevant context found in the knowledge base."
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"EXCERPT {i}:\n{chunk.strip()}")
        
        if debug:
            print(f"DEBUG: Formatted {len(context_parts)} excerpts for context")
        
        return "\n\n".join(context_parts)
    
    def check_knowledge_base(self) -> bool:
        """Check if knowledge base exists and is ready"""
        if self.is_indexed:
            return True
        return self._load_index()