"""
Artifact management for persona constitutions and related data.

This module handles the storage, retrieval, and management of persona artifacts
including compressed JSON storage, version management, and metadata tracking.
"""

import json
import gzip
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from ..models.persona_constitution import PersonaConstitution, StatisticalReport
from ...config.settings import Settings
from ...utils.logging import get_logger


class ArtifactManager:
    """
    Manages storage and retrieval of persona artifacts.
    
    Handles JSON serialization, compression, versioning, and metadata
    for persona constitutions and related data.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the artifact manager.
        
        Args:
            settings: Application settings containing storage configuration
        """
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Setup storage directories
        self.storage_root = Path("./data/storage")
        self.artifacts_dir = self.storage_root / "artifacts"
        self.personas_dir = self.storage_root / "personas"
        
        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persona registry
        self.registry_file = self.personas_dir / "persona_registry.json"
        self._ensure_registry()
    
    def _ensure_registry(self):
        """Ensure the persona registry file exists."""
        if not self.registry_file.exists():
            registry = {
                "personas": {},
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat()
            }
            self._save_json(registry, self.registry_file)
            self.logger.info("Created persona registry")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the persona registry."""
        try:
            return self._load_json(self.registry_file)
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
            return {"personas": {}, "created_at": datetime.utcnow().isoformat()}
    
    def _save_registry(self, registry: Dict[str, Any]):
        """Save the persona registry."""
        registry["last_updated"] = datetime.utcnow().isoformat()
        self._save_json(registry, self.registry_file)
    
    def _save_json(self, data: Union[Dict, List], file_path: Path, compress: bool = False):
        """Save data as JSON with optional compression."""
        try:
            if compress:
                # Save as compressed JSON
                with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                # Save as regular JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")
            raise
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data with automatic compression detection."""
        try:
            # Try compressed version first
            gz_path = Path(f"{file_path}.gz")
            if gz_path.exists():
                with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            
            # Try regular JSON
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            raise FileNotFoundError(f"No JSON file found at {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            raise
    
    def save_persona_constitution(self, persona: PersonaConstitution, name: str, compress: bool = True) -> str:
        """
        Save a persona constitution.
        
        Args:
            persona: PersonaConstitution object to save
            name: Name identifier for the persona
            compress: Whether to compress the saved file
            
        Returns:
            Artifact ID of the saved persona
        """
        # Create persona directory
        persona_dir = self.personas_dir / name
        persona_dir.mkdir(exist_ok=True)
        
        # Create artifacts subdirectory
        artifacts_dir = persona_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Generate artifact ID and filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_id = f"persona_{name}_{timestamp}"
        filename = f"{artifact_id}.json"
        
        # Save the persona constitution
        file_path = artifacts_dir / filename
        persona_dict = persona.dict()
        self._save_json(persona_dict, file_path, compress=compress)
        
        # Update registry
        registry = self._load_registry()
        if name not in registry["personas"]:
            registry["personas"][name] = {
                "created_at": datetime.utcnow().isoformat(),
                "artifacts": []
            }
        
        # Add artifact entry
        artifact_entry = {
            "artifact_id": artifact_id,
            "filename": filename + (".gz" if compress else ""),
            "created_at": datetime.utcnow().isoformat(),
            "compressed": compress,
            "file_path": str(file_path) + (".gz" if compress else ""),
            "quality_score": persona.overall_quality_score,
            "completeness_score": persona.completeness_score
        }
        
        registry["personas"][name]["artifacts"].append(artifact_entry)
        registry["personas"][name]["latest_artifact"] = artifact_id
        registry["personas"][name]["last_updated"] = datetime.utcnow().isoformat()
        
        self._save_registry(registry)
        
        self.logger.info(f"Saved persona constitution for '{name}' as {artifact_id}")
        return artifact_id
    
    def load_persona_constitution(self, name: str, artifact_id: Optional[str] = None) -> PersonaConstitution:
        """
        Load a persona constitution.
        
        Args:
            name: Name identifier for the persona
            artifact_id: Specific artifact ID to load (loads latest if None)
            
        Returns:
            PersonaConstitution object
        """
        registry = self._load_registry()
        
        if name not in registry["personas"]:
            raise ValueError(f"Persona '{name}' not found in registry")
        
        persona_info = registry["personas"][name]
        
        # Find the artifact to load
        if artifact_id is None:
            # Load latest artifact
            if not persona_info.get("artifacts"):
                raise ValueError(f"No artifacts found for persona '{name}'")
            artifact_entry = persona_info["artifacts"][-1]  # Latest artifact
        else:
            # Load specific artifact
            artifact_entry = None
            for entry in persona_info["artifacts"]:
                if entry["artifact_id"] == artifact_id:
                    artifact_entry = entry
                    break
            
            if artifact_entry is None:
                raise ValueError(f"Artifact '{artifact_id}' not found for persona '{name}'")
        
        # Load the artifact file
        file_path = Path(artifact_entry["file_path"])
        persona_dict = self._load_json(file_path.with_suffix(''))  # Remove .gz extension for _load_json
        
        # Create PersonaConstitution object
        persona = PersonaConstitution(**persona_dict)
        
        self.logger.info(f"Loaded persona constitution for '{name}' from {artifact_entry['artifact_id']}")
        return persona
    
    def list_personas(self) -> Dict[str, Dict[str, Any]]:
        """List all available personas with their metadata."""
        registry = self._load_registry()
        
        result = {}
        for name, info in registry["personas"].items():
            latest_artifact = None
            if info.get("artifacts"):
                latest_artifact = info["artifacts"][-1]
            
            result[name] = {
                "name": name,
                "created_at": info.get("created_at"),
                "last_updated": info.get("last_updated"),
                "artifact_count": len(info.get("artifacts", [])),
                "latest_quality_score": latest_artifact.get("quality_score") if latest_artifact else None,
                "latest_completeness_score": latest_artifact.get("completeness_score") if latest_artifact else None,
                "latest_artifact_id": info.get("latest_artifact")
            }
        
        return result
    
    def delete_persona(self, name: str) -> bool:
        """
        Delete a persona and all its artifacts.
        
        Args:
            name: Name identifier for the persona
            
        Returns:
            True if deleted successfully
        """
        registry = self._load_registry()
        
        if name not in registry["personas"]:
            return False
        
        # Delete artifacts directory
        persona_dir = self.personas_dir / name
        if persona_dir.exists():
            import shutil
            shutil.rmtree(persona_dir)
        
        # Remove from registry
        del registry["personas"][name]
        self._save_registry(registry)
        
        self.logger.info(f"Deleted persona '{name}' and all artifacts")
        return True
    
    def export_persona(self, name: str, output_path: Path, format: str = "json") -> Path:
        """
        Export a persona to a specific format.
        
        Args:
            name: Name identifier for the persona
            output_path: Output file path
            format: Export format ("json" or "yaml")
            
        Returns:
            Path to the exported file
        """
        persona = self.load_persona_constitution(name)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            self._save_json(persona.dict(), output_path, compress=False)
        elif format.lower() == "yaml":
            try:
                import yaml
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(persona.dict(), f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                raise ImportError("PyYAML is required for YAML export")
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported persona '{name}' to {output_path}")
        return output_path
    
    def get_persona_stats(self, name: str) -> Dict[str, Any]:
        """Get detailed statistics for a persona."""
        try:
            persona = self.load_persona_constitution(name)
            registry = self._load_registry()
            persona_info = registry["personas"].get(name, {})
            
            return {
                "name": name,
                "quality_summary": persona.get_quality_summary(),
                "artifact_count": len(persona_info.get("artifacts", [])),
                "created_at": persona_info.get("created_at"),
                "last_updated": persona_info.get("last_updated"),
                "mental_models_count": len(persona.mental_models),
                "core_beliefs_count": len(persona.core_beliefs),
                "has_statistical_report": persona.statistical_report is not None
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats for persona '{name}': {e}")
            return {"name": name, "error": str(e)}
    
    def save_statistical_report(self, report: StatisticalReport, persona_name: str) -> str:
        """Save a statistical analysis report."""
        persona_dir = self.personas_dir / persona_name
        persona_dir.mkdir(exist_ok=True)
        
        # Create analysis cache directory
        cache_dir = persona_dir / "analysis_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"statistical_report_{timestamp}.json"
        file_path = cache_dir / filename
        
        # Save report
        self._save_json(report.dict(), file_path, compress=True)
        
        self.logger.info(f"Saved statistical report for '{persona_name}'")
        return str(file_path)
    
    def load_latest_statistical_report(self, persona_name: str) -> Optional[StatisticalReport]:
        """Load the latest statistical report for a persona."""
        cache_dir = self.personas_dir / persona_name / "analysis_cache"
        
        if not cache_dir.exists():
            return None
        
        # Find latest report file
        report_files = list(cache_dir.glob("statistical_report_*.json*"))
        if not report_files:
            return None
        
        # Sort by modification time and get latest
        latest_file = max(report_files, key=lambda p: p.stat().st_mtime)
        
        try:
            report_dict = self._load_json(latest_file.with_suffix(''))  # Remove .gz extension
            return StatisticalReport(**report_dict)
        except Exception as e:
            self.logger.error(f"Failed to load statistical report: {e}")
            return None