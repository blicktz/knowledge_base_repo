"""
Extractor Cache Manager for persisting map-reduce extraction results
Enables resumable persona extraction by caching batch results and final consolidations
"""

import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from ..data.models.persona_constitution import MentalModel, CoreBelief, LinguisticStyle
from ..config.settings import Settings
from ..utils.logging import get_logger


class ExtractorCacheManager:
    """
    Manages caching of map-reduce extraction results to enable resumable extraction
    and avoid reprocessing batches when extraction is interrupted
    """
    
    def __init__(self, settings: Settings, persona_id: str):
        """
        Initialize the extractor cache manager
        
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
        
        # Setup subdirectories
        self.batch_results_dir = self.cache_dir / "batch_results"
        self.consolidated_dir = self.cache_dir / "consolidated"
        self.batch_results_dir.mkdir(exist_ok=True)
        self.consolidated_dir.mkdir(exist_ok=True)
        
        # Cache settings
        self.compression_enabled = settings.map_reduce_extraction.cache_compression
        self.cache_ttl_hours = settings.map_reduce_extraction.cache_ttl_hours
        
        # Cache files
        self.metadata_file = self.cache_dir / "extraction_metadata.json"
        self.progress_file = self.cache_dir / "extraction_progress.json"
    
    def _get_cache_directory(self) -> Path:
        """Get the cache directory path for this persona"""
        if hasattr(self.settings, 'get_persona_base_path'):
            # Multi-tenant mode
            persona_base = Path(self.settings.get_persona_base_path(self.persona_id))
            return persona_base / "map_reduce_cache"
        else:
            # Legacy mode
            base_cache = Path(self.settings.get_cache_path())
            return base_cache / "map_reduce" / self.persona_id
    
    def _calculate_batch_hash(self, documents: List[Dict[str, Any]], 
                             extraction_type: str) -> str:
        """
        Calculate a hash for a batch of documents and extraction type
        
        Args:
            documents: List of documents in the batch
            extraction_type: 'mental_models' or 'core_beliefs'
            
        Returns:
            SHA256 hash of batch content
        """
        # Sort documents by source to ensure consistent hashing
        sorted_docs = sorted(documents, key=lambda d: d.get('source', ''))
        
        # Combine content with extraction type
        combined_content = f"extraction_type:{extraction_type}\n"
        for doc in sorted_docs:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            combined_content += f"{source}:{content}\n"
        
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    def _calculate_full_corpus_hash(self, all_documents: List[Dict[str, Any]]) -> str:
        """Calculate hash for the entire corpus"""
        sorted_docs = sorted(all_documents, key=lambda d: d.get('source', ''))
        combined_content = ""
        for doc in sorted_docs:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            combined_content += f"{source}:{content}\n"
        
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
    
    def _get_batch_cache_file(self, batch_hash: str, extraction_type: str) -> Path:
        """Get the cache file path for a batch result"""
        filename = f"{extraction_type}_batch_{batch_hash[:16]}.json"
        if self.compression_enabled:
            filename += ".gz"
        return self.batch_results_dir / filename
    
    def _get_consolidated_cache_file(self, corpus_hash: str, extraction_type: str) -> Path:
        """Get the cache file path for consolidated results"""
        filename = f"{extraction_type}_consolidated_{corpus_hash[:16]}.json"
        if self.compression_enabled:
            filename += ".gz"
        return self.consolidated_dir / filename
    
    def save_batch_result(self, batch_documents: List[Dict[str, Any]], 
                         extraction_type: str, 
                         results: Union[List[MentalModel], List[CoreBelief]],
                         statistical_insights: str,
                         batch_index: int) -> str:
        """
        Save batch extraction results to cache
        
        Args:
            batch_documents: Documents in this batch
            extraction_type: 'mental_models' or 'core_beliefs'
            results: List of extracted models or beliefs
            statistical_insights: Statistical insights used
            batch_index: Index of this batch in the overall processing
            
        Returns:
            Cache file path
        """
        batch_hash = self._calculate_batch_hash(batch_documents, extraction_type)
        cache_file = self._get_batch_cache_file(batch_hash, extraction_type)
        
        # Prepare cache data
        cache_data = {
            "metadata": {
                "batch_hash": batch_hash,
                "batch_index": batch_index,
                "extraction_type": extraction_type,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(batch_documents),
                "total_words": sum(len(doc.get('content', '').split()) for doc in batch_documents),
                "extractor_version": "2.0.0",  # Map-reduce version
                "model_used": self.settings.map_reduce_extraction.map_phase_model
            },
            "batch_documents": [
                {
                    "source": doc.get('source', 'unknown'),
                    "word_count": len(doc.get('content', '').split()),
                    "char_count": len(doc.get('content', ''))
                }
                for doc in batch_documents
            ],
            "statistical_insights": statistical_insights,
            "results": [
                result.dict() if hasattr(result, 'dict') else result 
                for result in results
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
            
            self.logger.debug(f"Saved batch {batch_index} cache: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save batch cache: {e}")
            raise
    
    def load_batch_result(self, batch_documents: List[Dict[str, Any]], 
                         extraction_type: str) -> Optional[List[Union[MentalModel, CoreBelief]]]:
        """
        Load cached batch extraction results
        
        Args:
            batch_documents: Documents in the batch
            extraction_type: 'mental_models' or 'core_beliefs'
            
        Returns:
            List of extracted models/beliefs if valid cache found, None otherwise
        """
        batch_hash = self._calculate_batch_hash(batch_documents, extraction_type)
        cache_file = self._get_batch_cache_file(batch_hash, extraction_type)
        
        if not cache_file.exists():
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
            if not self._validate_batch_cache(cache_data, batch_documents, extraction_type):
                return None
            
            # Convert to appropriate objects
            results = []
            for result_data in cache_data["results"]:
                if extraction_type == "mental_models":
                    results.append(MentalModel.parse_obj(result_data))
                elif extraction_type == "core_beliefs":
                    results.append(CoreBelief.parse_obj(result_data))
            
            cache_timestamp = cache_data["metadata"]["timestamp"]
            batch_index = cache_data["metadata"].get("batch_index", "unknown")
            self.logger.debug(f"Loaded cached batch {batch_index} from {cache_timestamp}")
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Failed to load batch cache: {e}")
            return None
    
    def save_consolidated_result(self, all_documents: List[Dict[str, Any]], 
                               extraction_type: str,
                               consolidated_results: Union[List[MentalModel], List[CoreBelief]],
                               consolidation_metadata: Dict[str, Any]) -> str:
        """
        Save consolidated extraction results to cache
        
        Args:
            all_documents: All documents in the corpus
            extraction_type: 'mental_models' or 'core_beliefs'
            consolidated_results: Final consolidated results
            consolidation_metadata: Metadata about the consolidation process
            
        Returns:
            Cache file path
        """
        corpus_hash = self._calculate_full_corpus_hash(all_documents)
        cache_file = self._get_consolidated_cache_file(corpus_hash, extraction_type)
        
        # Prepare cache data
        cache_data = {
            "metadata": {
                "corpus_hash": corpus_hash,
                "extraction_type": extraction_type,
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(all_documents),
                "total_words": sum(len(doc.get('content', '').split()) for doc in all_documents),
                "consolidation_strategy": consolidation_metadata.get("strategy", "unknown"),
                "results_count": len(consolidated_results),
                "extractor_version": "2.0.0",
                "model_used": self.settings.map_reduce_extraction.reduce_phase_model
            },
            "consolidation_metadata": consolidation_metadata,
            "results": [
                result.dict() if hasattr(result, 'dict') else result 
                for result in consolidated_results
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
            
            self.logger.info(f"Saved consolidated {extraction_type} cache: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save consolidated cache: {e}")
            raise
    
    def load_consolidated_result(self, all_documents: List[Dict[str, Any]], 
                               extraction_type: str) -> Optional[List[Union[MentalModel, CoreBelief]]]:
        """
        Load cached consolidated extraction results
        
        Args:
            all_documents: All documents in the corpus
            extraction_type: 'mental_models' or 'core_beliefs'
            
        Returns:
            List of consolidated models/beliefs if valid cache found, None otherwise
        """
        corpus_hash = self._calculate_full_corpus_hash(all_documents)
        cache_file = self._get_consolidated_cache_file(corpus_hash, extraction_type)
        
        if not cache_file.exists():
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
            if not self._validate_consolidated_cache(cache_data, all_documents, extraction_type):
                return None
            
            # Convert to appropriate objects
            results = []
            for result_data in cache_data["results"]:
                if extraction_type == "mental_models":
                    results.append(MentalModel.parse_obj(result_data))
                elif extraction_type == "core_beliefs":
                    results.append(CoreBelief.parse_obj(result_data))
            
            cache_timestamp = cache_data["metadata"]["timestamp"]
            self.logger.info(f"Loaded consolidated {extraction_type} cache from {cache_timestamp}")
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Failed to load consolidated cache: {e}")
            return None
    
    def _validate_batch_cache(self, cache_data: Dict[str, Any], 
                            batch_documents: List[Dict[str, Any]], 
                            extraction_type: str) -> bool:
        """Validate if cached batch result is still valid"""
        metadata = cache_data.get("metadata", {})
        
        # Check extraction type
        if metadata.get("extraction_type") != extraction_type:
            return False
        
        # Check content hash
        current_hash = self._calculate_batch_hash(batch_documents, extraction_type)
        cached_hash = metadata.get("batch_hash")
        if current_hash != cached_hash:
            return False
        
        # Check document count
        if len(batch_documents) != metadata.get("document_count", 0):
            return False
        
        # Check age
        try:
            cache_time = datetime.fromisoformat(metadata.get("timestamp", ""))
            age = datetime.now() - cache_time
            if age > timedelta(hours=self.cache_ttl_hours):
                self.logger.debug(f"Batch cache expired: {age}")
                return False
        except Exception:
            return False
        
        return True
    
    def _validate_consolidated_cache(self, cache_data: Dict[str, Any], 
                                   all_documents: List[Dict[str, Any]], 
                                   extraction_type: str) -> bool:
        """Validate if cached consolidated result is still valid"""
        metadata = cache_data.get("metadata", {})
        
        # Check extraction type
        if metadata.get("extraction_type") != extraction_type:
            return False
        
        # Check corpus hash
        current_hash = self._calculate_full_corpus_hash(all_documents)
        cached_hash = metadata.get("corpus_hash")
        if current_hash != cached_hash:
            return False
        
        # Check document count
        if len(all_documents) != metadata.get("total_documents", 0):
            return False
        
        # Check age
        try:
            cache_time = datetime.fromisoformat(metadata.get("timestamp", ""))
            age = datetime.now() - cache_time
            if age > timedelta(hours=self.cache_ttl_hours):
                self.logger.debug(f"Consolidated cache expired: {age}")
                return False
        except Exception:
            return False
        
        return True
    
    def save_extraction_progress(self, total_batches: int, completed_batches: List[int], 
                               extraction_type: str, corpus_hash: str):
        """Save extraction progress for resuming interrupted extractions"""
        progress_data = {
            "corpus_hash": corpus_hash,
            "extraction_type": extraction_type,
            "total_batches": total_batches,
            "completed_batches": completed_batches,
            "last_updated": datetime.now().isoformat(),
            "completion_percentage": len(completed_batches) / total_batches * 100
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save extraction progress: {e}")
    
    def load_extraction_progress(self, corpus_hash: str, 
                               extraction_type: str) -> Optional[Dict[str, Any]]:
        """Load extraction progress for resuming"""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Validate progress data
            if (progress_data.get("corpus_hash") == corpus_hash and 
                progress_data.get("extraction_type") == extraction_type):
                return progress_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load extraction progress: {e}")
        
        return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached extractions"""
        batch_files = list(self.batch_results_dir.glob("*.json*"))
        consolidated_files = list(self.consolidated_dir.glob("*.json*"))
        
        cache_info = {
            "status": "available" if (batch_files or consolidated_files) else "no_cache",
            "cache_dir": str(self.cache_dir),
            "batch_results": {
                "count": len(batch_files),
                "files": [
                    {
                        "file": f.name,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in batch_files
                ]
            },
            "consolidated_results": {
                "count": len(consolidated_files),
                "files": [
                    {
                        "file": f.name,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in consolidated_files
                ]
            }
        }
        
        # Add progress info if available
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                cache_info["last_progress"] = progress_data
            except Exception:
                pass
        
        return cache_info
    
    def create_batch_log_directory(self, batch_hash: str, extraction_type: str) -> Path:
        """
        Create directory for batch processing logs
        
        Args:
            batch_hash: Hash of the batch content
            extraction_type: Type of extraction ('mental_models' or 'core_beliefs')
            
        Returns:
            Path to the batch log directory
        """
        batch_dir = self.cache_dir / "xml_responses" / extraction_type / f"batch_{batch_hash[:16]}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir
    
    def save_batch_input(self, batch_dir: Path, prompt: str) -> Path:
        """
        Save the complete prompt sent to LLM
        
        Args:
            batch_dir: Directory for this batch's logs
            prompt: The complete prompt sent to the LLM
            
        Returns:
            Path to the saved input file
        """
        input_file = batch_dir / "input.txt"
        try:
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            self.logger.debug(f"Saved batch input to {input_file}")
            return input_file
        except Exception as e:
            self.logger.error(f"Failed to save batch input: {e}")
            raise
    
    def save_batch_response(self, batch_dir: Path, response: str) -> Path:
        """
        Save the complete LLM response
        
        Args:
            batch_dir: Directory for this batch's logs
            response: The complete response from the LLM
            
        Returns:
            Path to the saved response file
        """
        response_file = batch_dir / "response.xml"
        try:
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)
            self.logger.debug(f"Saved batch response to {response_file}")
            return response_file
        except Exception as e:
            self.logger.error(f"Failed to save batch response: {e}")
            raise
    
    def save_batch_output(self, batch_dir: Path, output_json: list) -> Path:
        """
        Save the extracted JSON output
        
        Args:
            batch_dir: Directory for this batch's logs
            output_json: The extracted JSON output
            
        Returns:
            Path to the saved output file
        """
        output_file = batch_dir / "output.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved batch output to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Failed to save batch output: {e}")
            raise
    
    def save_batch_metadata(self, batch_dir: Path, metadata: dict) -> Path:
        """
        Save batch processing metadata
        
        Args:
            batch_dir: Directory for this batch's logs
            metadata: Metadata about the batch processing
            
        Returns:
            Path to the saved metadata file
        """
        metadata_file = batch_dir / "metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved batch metadata to {metadata_file}")
            return metadata_file
        except Exception as e:
            self.logger.error(f"Failed to save batch metadata: {e}")
            raise
    
    def get_batch_log(self, batch_hash: str, extraction_type: str) -> Dict[str, Any]:
        """
        Retrieve all logged data for a batch
        
        Args:
            batch_hash: Hash of the batch content
            extraction_type: Type of extraction
            
        Returns:
            Dictionary containing all batch log data
        """
        batch_dir = self.cache_dir / "xml_responses" / extraction_type / f"batch_{batch_hash[:16]}"
        
        log_data = {}
        
        # Read input
        input_file = batch_dir / "input.txt"
        if input_file.exists():
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    log_data['input'] = f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read input file: {e}")
        
        # Read response
        response_file = batch_dir / "response.xml"
        if response_file.exists():
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    log_data['response'] = f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read response file: {e}")
        
        # Read output
        output_file = batch_dir / "output.json"
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    log_data['output'] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read output file: {e}")
        
        # Read metadata
        metadata_file = batch_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    log_data['metadata'] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read metadata file: {e}")
        
        return log_data
    
    def save_linguistic_style(self, all_documents: List[Dict[str, Any]], 
                             statistical_insights: str, linguistic_style: LinguisticStyle):
        """
        Save linguistic style extraction result to cache
        
        Args:
            all_documents: All documents used for extraction
            statistical_insights: Statistical insights used
            linguistic_style: Extracted linguistic style result
        """
        cache_key = self._calculate_linguistic_style_hash(all_documents, statistical_insights)
        cache_file = self.cache_dir / f"linguistic_style_{cache_key}.json"
        
        try:
            # Prepare cache data
            cache_data = {
                "linguistic_style": linguistic_style.model_dump() if hasattr(linguistic_style, 'model_dump') else linguistic_style.__dict__,
                "metadata": {
                    "corpus_hash": self._calculate_full_corpus_hash(all_documents),
                    "insights_hash": hashlib.sha256(statistical_insights.encode('utf-8')).hexdigest(),
                    "total_documents": len(all_documents),
                    "timestamp": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }
            }
            
            # Write to cache file
            if self.compression_enabled:
                with gzip.open(f"{cache_file}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
            else:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                    
            self.logger.debug(f"Saved linguistic style to cache: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save linguistic style cache: {e}")
    
    def load_linguistic_style(self, all_documents: List[Dict[str, Any]], 
                             statistical_insights: str) -> Optional[LinguisticStyle]:
        """
        Load cached linguistic style extraction result
        
        Args:
            all_documents: All documents used for extraction
            statistical_insights: Statistical insights used
            
        Returns:
            Cached linguistic style if valid cache found, None otherwise
        """
        cache_key = self._calculate_linguistic_style_hash(all_documents, statistical_insights)
        cache_file = self.cache_dir / f"linguistic_style_{cache_key}.json"
        
        # Check both compressed and uncompressed versions
        compressed_file = Path(f"{cache_file}.gz")
        
        target_file = None
        if compressed_file.exists():
            target_file = compressed_file
        elif cache_file.exists():
            target_file = cache_file
        else:
            return None
        
        try:
            # Load cache data
            if target_file.suffix == '.gz':
                with gzip.open(target_file, 'rt', encoding='utf-8') as f:
                    cache_data = json.load(f)
            else:
                with open(target_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Validate cache
            if not self._validate_linguistic_style_cache(cache_data, all_documents, statistical_insights):
                self.logger.debug(f"Linguistic style cache validation failed: {cache_key}")
                return None
            
            # Reconstruct LinguisticStyle object
            style_data = cache_data.get("linguistic_style", {})
            linguistic_style = LinguisticStyle(
                tone=style_data.get('tone', ''),
                catchphrases=style_data.get('catchphrases', []),
                vocabulary=style_data.get('vocabulary', []),
                sentence_structures=style_data.get('sentence_structures', []),
                communication_style=style_data.get('communication_style', {})
            )
            
            self.logger.debug(f"Loaded linguistic style from cache: {cache_key}")
            return linguistic_style
            
        except Exception as e:
            self.logger.warning(f"Failed to load linguistic style cache: {e}")
            return None
    
    def _calculate_linguistic_style_hash(self, all_documents: List[Dict[str, Any]], 
                                        statistical_insights: str) -> str:
        """Calculate hash for linguistic style cache key"""
        corpus_hash = self._calculate_full_corpus_hash(all_documents)
        insights_hash = hashlib.sha256(statistical_insights.encode('utf-8')).hexdigest()
        
        # Combine corpus and insights for final hash
        combined = f"linguistic_style:{corpus_hash}:{insights_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def _validate_linguistic_style_cache(self, cache_data: Dict[str, Any], 
                                        all_documents: List[Dict[str, Any]], 
                                        statistical_insights: str) -> bool:
        """Validate if cached linguistic style result is still valid"""
        metadata = cache_data.get("metadata", {})
        
        # Check corpus hash
        current_corpus_hash = self._calculate_full_corpus_hash(all_documents)
        cached_corpus_hash = metadata.get("corpus_hash")
        if current_corpus_hash != cached_corpus_hash:
            return False
        
        # Check insights hash
        current_insights_hash = hashlib.sha256(statistical_insights.encode('utf-8')).hexdigest()
        cached_insights_hash = metadata.get("insights_hash")
        if current_insights_hash != cached_insights_hash:
            return False
        
        # Check document count
        if len(all_documents) != metadata.get("total_documents", 0):
            return False
        
        # Check age
        try:
            cache_time = datetime.fromisoformat(metadata.get("timestamp", ""))
            age = datetime.now() - cache_time
            if age > timedelta(hours=self.cache_ttl_hours):
                self.logger.debug(f"Linguistic style cache expired: {age}")
                return False
        except Exception:
            return False
        
        return True
    
    def clear_cache(self, extraction_type: Optional[str] = None, 
                   older_than_hours: Optional[int] = None):
        """
        Clear cached extraction files
        
        Args:
            extraction_type: Only clear specific type ('mental_models', 'core_beliefs', None = all)
            older_than_hours: Only clear files older than this many hours (None = all)
        """
        cleared_count = 0
        
        # Clear batch results
        pattern = f"{extraction_type}_*.json*" if extraction_type else "*.json*"
        for cache_file in self.batch_results_dir.glob(pattern):
            should_delete = True
            
            if older_than_hours is not None:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                should_delete = file_age.total_seconds() > (older_than_hours * 3600)
            
            if should_delete:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        # Clear consolidated results
        for cache_file in self.consolidated_dir.glob(pattern):
            should_delete = True
            
            if older_than_hours is not None:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                should_delete = file_age.total_seconds() > (older_than_hours * 3600)
            
            if should_delete:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        # Clear progress file if no type specified
        if extraction_type is None and self.progress_file.exists():
            try:
                self.progress_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete progress file: {e}")
        
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} extraction cache files")