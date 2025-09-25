"""Persona Data Tool - Loads and extracts static persona data from latest artifact"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from .base_tool import BasePersonaTool
from ..config.settings import Settings
from ..utils.artifact_discovery import ArtifactDiscovery
from ..utils.logging import get_logger


class PersonaDataTool(BasePersonaTool):
    """Loads and extracts static persona data from latest artifact"""
    
    name: str = "persona_data"
    description: str = "Extract linguistic style and static persona information"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        object.__setattr__(self, 'artifact_discovery', ArtifactDiscovery(settings))
        object.__setattr__(self, '_cached_data', None)
        object.__setattr__(self, '_cache_timestamp', None)
        
    def execute(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract:
        - linguistic_style: Tone, catchphrases, vocabulary
        - communication_patterns: Speaking patterns and style
        - persona_metadata: Basic persona information
        """
        self.logger.info(f"Loading persona data for: {self.persona_id}")
        
        # Check cache validity
        cache_ttl_minutes = self.settings.agent.tools.persona_data.cache_ttl_minutes
        if self._cached_data and self._is_cache_valid_minutes(cache_ttl_minutes):
            self.logger.info("Using cached persona data")
            return self._cached_data
        
        # Auto-discover and load latest artifact
        json_path, artifact_info = self.artifact_discovery.get_latest_artifact_json(self.persona_id)
        
        self.logger.info(f"Loading from artifact: {artifact_info.file_path.name}")
        
        # Extract relevant persona data
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        # Extract only linguistic style and static data
        extracted_data = self._extract_persona_data(full_data, artifact_info)
        
        # Cache the data
        self._cached_data = extracted_data
        self._cache_timestamp = datetime.now()
        
        # Clean up temp file if needed
        if self.settings.agent.tools.persona_data.auto_cleanup_temp_files:
            self.artifact_discovery.cleanup_temp_file(json_path)
        
        self.logger.info("Persona data extraction completed")
        
        return extracted_data
    
    def _extract_persona_data(self, full_data: Dict, artifact_info: Any) -> Dict[str, Any]:
        """Extract relevant fields from full persona data"""
        
        # Extract linguistic style
        linguistic_style = full_data.get('linguistic_style', {})
        if not linguistic_style:
            # Try alternative field names
            linguistic_style = full_data.get('writing_style', {})
            if not linguistic_style:
                linguistic_style = full_data.get('communication_style', {})
        
        # Extract communication patterns
        communication_patterns = full_data.get('communication_patterns', {})
        if not communication_patterns:
            # Build from available data
            communication_patterns = {
                'tone': full_data.get('tone', 'professional'),
                'formality': full_data.get('formality', 'formal'),
                'typical_phrases': full_data.get('catchphrases', [])
            }
        
        # Build persona metadata
        persona_metadata = {
            'name': full_data.get('name', self.persona_id),
            'description': full_data.get('description', ''),
            'expertise': full_data.get('expertise', []),
            'extraction_timestamp': artifact_info.timestamp.isoformat(),
            'artifact_source': str(artifact_info.file_path.name)
        }
        
        # Add any catchphrases or signature elements
        if 'catchphrases' in full_data:
            linguistic_style['catchphrases'] = full_data['catchphrases']
        
        if 'vocabulary' in full_data:
            linguistic_style['vocabulary'] = full_data['vocabulary']
        
        return {
            'linguistic_style': linguistic_style,
            'communication_patterns': communication_patterns,
            'persona_metadata': persona_metadata
        }
    
    def _is_cache_valid_minutes(self, ttl_minutes: int) -> bool:
        """Check if cache is valid based on minutes TTL"""
        if not self._cache_timestamp:
            return False
        
        age_minutes = (datetime.now() - self._cache_timestamp).total_seconds() / 60
        return age_minutes < ttl_minutes