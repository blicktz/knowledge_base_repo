"""
Artifact Discovery Utility

Auto-discovers and processes the latest Phase 1 JSON artifacts for personas.
Uses configuration-based paths and handles automatic decompression.
"""

import gzip
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..config.settings import Settings
from ..utils.logging import get_logger


@dataclass
class ArtifactInfo:
    """Information about a discovered artifact."""
    file_path: Path
    persona_id: str
    timestamp: datetime
    file_size_bytes: int
    is_compressed: bool


class ArtifactDiscoveryError(Exception):
    """Raised when artifact discovery fails."""
    pass


class ArtifactDiscovery:
    """
    Discovers and processes persona artifacts automatically.
    
    Finds the latest Phase 1 JSON artifact for a persona, handles
    decompression if needed, and returns the path to processable JSON.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize artifact discovery.
        
        Args:
            settings: Settings instance for configuration
        """
        self.settings = settings or Settings.from_default_config()
        self.logger = get_logger(__name__)
        
        # Get base storage directory from config
        self.base_storage_dir = Path(self.settings.storage.base_storage_dir)
        self.logger.debug(f"Artifact discovery using base path: {self.base_storage_dir}")
    
    def get_persona_artifacts_dir(self, persona_id: str) -> Path:
        """
        Get the artifacts directory for a specific persona.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            Path to persona artifacts directory
        """
        return self.base_storage_dir / "personas" / persona_id / "artifacts"
    
    def discover_artifacts(self, persona_id: str) -> List[ArtifactInfo]:
        """
        Discover all artifacts for a persona.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            List of ArtifactInfo objects sorted by timestamp (newest first)
            
        Raises:
            ArtifactDiscoveryError: If artifacts directory doesn't exist or no artifacts found
        """
        artifacts_dir = self.get_persona_artifacts_dir(persona_id)
        
        if not artifacts_dir.exists():
            raise ArtifactDiscoveryError(
                f"Artifacts directory does not exist: {artifacts_dir}"
            )
        
        self.logger.debug(f"Scanning for artifacts in: {artifacts_dir}")
        
        # Pattern for persona artifacts: persona_{persona_id}_YYYYMMDD_HHMMSS.json[.gz]
        pattern = re.compile(rf"^persona_{re.escape(persona_id)}_(\d{{8}}_\d{{6}})\.json(\.gz)?$")
        
        artifacts = []
        for file_path in artifacts_dir.iterdir():
            if not file_path.is_file():
                continue
                
            match = pattern.match(file_path.name)
            if not match:
                continue
            
            timestamp_str = match.group(1)
            is_compressed = match.group(2) is not None
            
            try:
                # Parse timestamp from filename
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                artifact_info = ArtifactInfo(
                    file_path=file_path,
                    persona_id=persona_id,
                    timestamp=timestamp,
                    file_size_bytes=file_path.stat().st_size,
                    is_compressed=is_compressed
                )
                artifacts.append(artifact_info)
                
            except ValueError as e:
                self.logger.warning(f"Could not parse timestamp from {file_path.name}: {e}")
                continue
        
        if not artifacts:
            raise ArtifactDiscoveryError(
                f"No artifacts found for persona '{persona_id}' in {artifacts_dir}"
            )
        
        # Sort by timestamp, newest first
        artifacts.sort(key=lambda x: x.timestamp, reverse=True)
        
        self.logger.info(f"Found {len(artifacts)} artifacts for persona '{persona_id}'")
        return artifacts
    
    def get_latest_artifact(self, persona_id: str) -> ArtifactInfo:
        """
        Get the latest artifact for a persona.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            ArtifactInfo for the latest artifact
            
        Raises:
            ArtifactDiscoveryError: If no artifacts found
        """
        artifacts = self.discover_artifacts(persona_id)
        latest = artifacts[0]  # Already sorted newest first
        
        self.logger.info(
            f"Latest artifact for '{persona_id}': {latest.file_path.name} "
            f"(timestamp: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        
        return latest
    
    def extract_artifact_json(self, artifact_info: ArtifactInfo) -> str:
        """
        Extract JSON content from artifact file.
        
        Args:
            artifact_info: Artifact information
            
        Returns:
            Path to extracted JSON file (temporary file if compressed)
            
        Raises:
            ArtifactDiscoveryError: If extraction fails
        """
        try:
            if artifact_info.is_compressed:
                # Create temporary file for extracted content
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    prefix=f'artifact_{artifact_info.persona_id}_',
                    delete=False
                )
                
                self.logger.debug(f"Extracting {artifact_info.file_path} to {temp_file.name}")
                
                # Decompress and write to temporary file
                with gzip.open(artifact_info.file_path, 'rt', encoding='utf-8') as gz_file:
                    content = gz_file.read()
                    temp_file.write(content)
                    temp_file.flush()
                
                temp_file.close()
                
                # Validate JSON structure
                self._validate_json_structure(temp_file.name)
                
                self.logger.info(
                    f"Extracted compressed artifact to: {temp_file.name} "
                    f"({len(content)} characters)"
                )
                
                return temp_file.name
            else:
                # File is already uncompressed JSON
                json_path = str(artifact_info.file_path)
                
                # Validate JSON structure
                self._validate_json_structure(json_path)
                
                self.logger.info(f"Using uncompressed artifact: {json_path}")
                return json_path
                
        except Exception as e:
            raise ArtifactDiscoveryError(f"Failed to extract artifact JSON: {e}")
    
    def _validate_json_structure(self, json_path: str) -> None:
        """
        Validate that the JSON file has the expected Phase 1 artifact structure.
        
        Args:
            json_path: Path to JSON file
            
        Raises:
            ArtifactDiscoveryError: If JSON structure is invalid
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check for expected top-level structure
            if not isinstance(data, dict):
                raise ArtifactDiscoveryError("Artifact JSON must be a dictionary")
            
            # Check for mental_models array
            mental_models = data.get('mental_models')
            if mental_models is not None and not isinstance(mental_models, list):
                raise ArtifactDiscoveryError("mental_models must be an array")
            
            # Check for core_beliefs array
            core_beliefs = data.get('core_beliefs')
            if core_beliefs is not None and not isinstance(core_beliefs, list):
                raise ArtifactDiscoveryError("core_beliefs must be an array")
            
            # Log structure info
            mm_count = len(mental_models) if mental_models else 0
            cb_count = len(core_beliefs) if core_beliefs else 0
            
            self.logger.debug(
                f"Validated artifact JSON: {mm_count} mental models, "
                f"{cb_count} core beliefs"
            )
            
        except json.JSONDecodeError as e:
            raise ArtifactDiscoveryError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ArtifactDiscoveryError(f"JSON validation failed: {e}")
    
    def get_latest_artifact_json(self, persona_id: str) -> Tuple[str, ArtifactInfo]:
        """
        Get the latest artifact JSON file path for a persona.
        
        This is the main method that combines discovery and extraction.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            Tuple of (json_file_path, artifact_info)
            
        Raises:
            ArtifactDiscoveryError: If discovery or extraction fails
        """
        self.logger.info(f"Auto-discovering latest artifact for persona: {persona_id}")
        
        # Discover latest artifact
        latest_artifact = self.get_latest_artifact(persona_id)
        
        # Extract JSON content
        json_path = self.extract_artifact_json(latest_artifact)
        
        self.logger.info(
            f"Successfully prepared artifact JSON: {json_path} "
            f"(from {latest_artifact.file_path.name})"
        )
        
        return json_path, latest_artifact
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary extracted file.
        
        Args:
            file_path: Path to temporary file to clean up
        """
        try:
            temp_path = Path(file_path)
            if temp_path.exists() and temp_path.parent == Path(tempfile.gettempdir()):
                temp_path.unlink()
                self.logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


def create_artifact_discovery(settings: Optional[Settings] = None) -> ArtifactDiscovery:
    """
    Factory function to create ArtifactDiscovery instance.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        ArtifactDiscovery instance
    """
    return ArtifactDiscovery(settings)