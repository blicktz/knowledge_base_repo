"""
Analysis Cache Manager for persisting statistical analysis results
Enables fast persona extraction iterations by caching expensive analysis steps
"""

import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from ..data.models.persona_constitution import StatisticalReport
from ..config.settings import Settings
from ..utils.logging import get_logger


class AnalysisCacheManager:
    """
    Manages caching of statistical analysis results to avoid recomputing
    expensive analysis steps during persona extraction iterations
    """
    
    def __init__(self, settings: Settings, persona_id: str):
        """
        Initialize the analysis cache manager
        
        Args:
            settings: Application settings
            persona_id: Identifier for the persona (for multi-tenant isolation)
        """
        self.settings = settings
        self.persona_id = persona_id
        self.logger = get_logger(__name__)
        
        # Setup cache directory
        self.cache_dir = self._get_cache_directory()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.compression_enabled = settings.storage.compression.get('enabled', True)
        self.max_cache_age_days = settings.performance.cache_size  # Reuse this setting
        
        # Cache files
        self.metadata_file = self.cache_dir / "metadata.json"
        self.latest_link = self.cache_dir / "latest"
    
    def _get_cache_directory(self) -> Path:
        """Get the cache directory path for this persona"""
        if hasattr(self.settings, 'get_persona_base_path'):
            # Multi-tenant mode
            persona_base = Path(self.settings.get_persona_base_path(self.persona_id))
            return persona_base / "analysis_cache"
        else:
            # Legacy mode
            base_cache = Path(self.settings.get_cache_path())
            return base_cache / "analysis" / self.persona_id
    
    def _calculate_document_hash(self, documents: List[Dict[str, Any]]) -> str:
        """
        Calculate a hash of the document contents for cache validation
        
        Args:
            documents: List of documents with 'content' key
            
        Returns:
            SHA256 hash of combined document contents
        """
        # Sort documents by source to ensure consistent hashing
        sorted_docs = sorted(documents, key=lambda d: d.get('source', ''))
        
        # Combine all content
        combined_content = ""
        for doc in sorted_docs:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            combined_content += f"{source}:{content}\n"
        
        # Calculate hash
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, content_hash: str) -> Path:
        """Get the cache file path for a given content hash"""
        filename = f"statistical_report_{content_hash[:16]}.json"
        if self.compression_enabled:
            filename += ".gz"
        return self.cache_dir / filename
    
    def _save_metadata(self, content_hash: str, documents: List[Dict[str, Any]], 
                      analysis_timestamp: str):
        """Save cache metadata"""
        metadata = {
            "latest_hash": content_hash,
            "timestamp": analysis_timestamp,
            "document_count": len(documents),
            "total_words": sum(len(doc.get('content', '').split()) for doc in documents),
            "sources": [doc.get('source', 'unknown') for doc in documents],
            "analyzer_version": "1.0.0",
            "settings_hash": self._calculate_settings_hash()
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")
    
    def _calculate_settings_hash(self) -> str:
        """Calculate hash of relevant settings that would affect analysis"""
        settings_data = {
            'spacy_model': self.settings.statistical_analysis.spacy.get('model'),
            'nltk_config': self.settings.statistical_analysis.nltk,
            'keywords_config': self.settings.statistical_analysis.keywords
        }
        settings_str = json.dumps(settings_data, sort_keys=True)
        return hashlib.sha256(settings_str.encode('utf-8')).hexdigest()[:16]
    
    def save_analysis(self, statistical_report: StatisticalReport, 
                     documents: List[Dict[str, Any]]) -> str:
        """
        Save statistical analysis results to cache
        
        Args:
            statistical_report: The analysis results to cache
            documents: Original documents that were analyzed
            
        Returns:
            Cache file path
        """
        content_hash = self._calculate_document_hash(documents)
        cache_file = self._get_cache_file_path(content_hash)
        
        # Prepare cache data
        cache_data = {
            "metadata": {
                "content_hash": content_hash,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(documents),
                "total_words": statistical_report.total_words,
                "analyzer_version": "1.0.0",
                "settings_hash": self._calculate_settings_hash()
            },
            "statistical_report": statistical_report.dict(),
            "document_summaries": [
                {
                    "source": doc.get('source', 'unknown'),
                    "word_count": len(doc.get('content', '').split()),
                    "char_count": len(doc.get('content', ''))
                }
                for doc in documents
            ]
        }
        
        try:
            # Save cache file
            if self.compression_enabled:
                with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
            else:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            self._save_metadata(content_hash, documents, cache_data["metadata"]["timestamp"])
            
            # Update latest link
            if self.latest_link.exists():
                self.latest_link.unlink()
            self.latest_link.symlink_to(cache_file.name)
            
            self.logger.info(f"Saved analysis cache: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis cache: {e}")
            raise
    
    def load_analysis(self, documents: List[Dict[str, Any]], 
                     max_age_hours: Optional[int] = None) -> Optional[StatisticalReport]:
        """
        Load cached statistical analysis results
        
        Args:
            documents: Documents to validate cache against
            max_age_hours: Maximum age of cache in hours (None = no limit)
            
        Returns:
            StatisticalReport if valid cache found, None otherwise
        """
        content_hash = self._calculate_document_hash(documents)
        cache_file = self._get_cache_file_path(content_hash)
        
        if not cache_file.exists():
            self.logger.debug(f"No cache file found for hash: {content_hash[:16]}")
            return None
        
        try:
            # Load cache data
            if str(cache_file).endswith('.gz'):
                with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                    cache_data = json.load(f)
            else:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Validate cache
            if not self._validate_cache(cache_data, documents, max_age_hours):
                return None
            
            # Convert to StatisticalReport object
            report_data = cache_data["statistical_report"]
            statistical_report = StatisticalReport.parse_obj(report_data)
            
            cache_timestamp = cache_data["metadata"]["timestamp"]
            self.logger.info(f"Loaded cached analysis from {cache_timestamp}")
            
            return statistical_report
            
        except Exception as e:
            self.logger.warning(f"Failed to load analysis cache: {e}")
            return None
    
    def _validate_cache(self, cache_data: Dict[str, Any], 
                       documents: List[Dict[str, Any]], 
                       max_age_hours: Optional[int]) -> bool:
        """
        Validate if cached analysis is still valid
        
        Args:
            cache_data: Loaded cache data
            documents: Current documents to validate against
            max_age_hours: Maximum age in hours
            
        Returns:
            True if cache is valid, False otherwise
        """
        metadata = cache_data.get("metadata", {})
        
        # Check content hash
        current_hash = self._calculate_document_hash(documents)
        cached_hash = metadata.get("content_hash")
        if current_hash != cached_hash:
            self.logger.debug("Cache invalid: content hash mismatch")
            return False
        
        # Check document count
        if len(documents) != metadata.get("document_count", 0):
            self.logger.debug("Cache invalid: document count mismatch")
            return False
        
        # Check age
        if max_age_hours is not None:
            try:
                cache_time = datetime.fromisoformat(metadata.get("timestamp", ""))
                age = datetime.now() - cache_time
                if age > timedelta(hours=max_age_hours):
                    self.logger.debug(f"Cache invalid: too old ({age})")
                    return False
            except Exception:
                self.logger.debug("Cache invalid: invalid timestamp")
                return False
        
        # Check settings compatibility
        current_settings_hash = self._calculate_settings_hash()
        cached_settings_hash = metadata.get("settings_hash")
        if cached_settings_hash and current_settings_hash != cached_settings_hash:
            self.logger.debug("Cache invalid: settings changed")
            return False
        
        return True
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached analyses
        
        Returns:
            Dictionary with cache information
        """
        if not self.metadata_file.exists():
            return {"status": "no_cache", "files": []}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # List cache files
            cache_files = []
            for cache_file in self.cache_dir.glob("statistical_report_*.json*"):
                if cache_file.is_file():
                    cache_files.append({
                        "file": cache_file.name,
                        "size": cache_file.stat().st_size,
                        "modified": datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
                    })
            
            return {
                "status": "available",
                "latest": metadata,
                "files": cache_files,
                "cache_dir": str(self.cache_dir)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get cache info: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached analysis files
        
        Args:
            older_than_days: Only clear files older than this many days (None = all)
        """
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("statistical_report_*.json*"):
            should_delete = True
            
            if older_than_days is not None:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                should_delete = file_age.days > older_than_days
            
            if should_delete:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        # Clear metadata if all files cleared
        if older_than_days is None and self.metadata_file.exists():
            try:
                self.metadata_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete metadata file: {e}")
        
        # Clear latest link if it exists and is broken
        if self.latest_link.exists() and not self.latest_link.resolve().exists():
            try:
                self.latest_link.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete latest link: {e}")
        
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} cache files")
    
    def has_valid_cache(self, documents: List[Dict[str, Any]], 
                       max_age_hours: Optional[int] = None) -> bool:
        """
        Check if valid cache exists for the given documents
        
        Args:
            documents: Documents to check cache for
            max_age_hours: Maximum age in hours
            
        Returns:
            True if valid cache exists, False otherwise
        """
        content_hash = self._calculate_document_hash(documents)
        cache_file = self._get_cache_file_path(content_hash)
        
        if not cache_file.exists():
            return False
        
        try:
            # Load and validate cache
            if str(cache_file).endswith('.gz'):
                with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                    cache_data = json.load(f)
            else:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            return self._validate_cache(cache_data, documents, max_age_hours)
            
        except Exception:
            return False