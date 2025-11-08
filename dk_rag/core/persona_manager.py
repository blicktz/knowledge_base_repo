"""
Persona Manager for multi-tenant architecture
Manages persona-specific vector stores and ensures data isolation
"""

import re
import json
import gzip
from typing import Dict, Optional, List, Any, Union
from pathlib import Path
from datetime import datetime, timezone

from ..data.storage.langchain_vector_store import LangChainVectorStore as VectorStore
from ..data.models.persona_constitution import PersonaConstitution, StatisticalReport
from ..config.settings import Settings
from ..utils.logging import get_logger


class PersonaManager:
    """
    Manages multiple personas with isolated vector stores and artifacts
    """
    
    def __init__(self, settings: Settings):
        """Initialize the persona manager"""
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Setup base personas directory
        self.personas_base_dir = Path(self.settings.get_personas_base_dir())
        self.personas_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded vector stores
        self.active_vector_stores: Dict[str, VectorStore] = {}
        
        # Load persona registry
        self.registry_path = self.personas_base_dir / "persona_registry.json"
        self.persona_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the persona registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    # Handle both old flat format and new nested format
                    if "personas" in data:
                        return data["personas"]
                    else:
                        return data
            except Exception as e:
                self.logger.error(f"Failed to load persona registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the persona registry to disk"""
        try:
            # Save in unified nested format
            registry_data = {
                "personas": self.persona_registry,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save persona registry: {e}")
    
    def _sanitize_persona_id(self, persona_name: str) -> str:
        """
        Create a safe persona ID from a name
        
        Args:
            persona_name: Human-readable persona name
            
        Returns:
            Sanitized persona ID suitable for file paths
        """
        # Convert to lowercase, replace spaces with underscores
        persona_id = persona_name.lower().strip()
        persona_id = re.sub(r'\s+', '_', persona_id)
        # Remove special characters, keep only alphanumeric and underscores
        persona_id = re.sub(r'[^a-z0-9_]', '', persona_id)
        # Remove consecutive underscores
        persona_id = re.sub(r'_+', '_', persona_id)
        # Remove leading/trailing underscores
        persona_id = persona_id.strip('_')
        
        return persona_id
    
    def register_persona(self, persona_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new persona in the system
        
        Args:
            persona_name: Human-readable persona name
            metadata: Optional metadata about the persona
            
        Returns:
            The persona ID
        """
        persona_id = self._sanitize_persona_id(persona_name)
        
        if persona_id in self.persona_registry:
            self.logger.info(f"Persona '{persona_name}' already registered as '{persona_id}'")
            return persona_id
        
        # Create persona entry with unified structure
        self.persona_registry[persona_id] = {
            "name": persona_name,
            "id": persona_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "stats": {
                "documents": 0,
                "chunks": 0,
                "stats_updated_at": None
            },
            "artifacts": [],
            "latest_artifact": None
        }
        
        # Create persona directories
        persona_dir = self.personas_base_dir / persona_id
        persona_dir.mkdir(parents=True, exist_ok=True)
        (persona_dir / "vector_db").mkdir(parents=True, exist_ok=True)
        (persona_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Registered new persona: '{persona_name}' as '{persona_id}'")
        return persona_id
    
    def get_persona_vector_store(self, persona_id: str) -> VectorStore:
        """
        Get or create a vector store for a specific persona

        Args:
            persona_id: The persona identifier

        Returns:
            Persona-specific vector store instance
        """
        # Check if persona is registered
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_id}' not registered. Register it first.")

        # Return cached instance if available
        if persona_id in self.active_vector_stores:
            return self.active_vector_stores[persona_id]

        # Get persona language
        language = self.get_persona_language(persona_id)

        # Create persona-specific settings
        persona_settings = self._create_persona_settings(persona_id)

        # Initialize persona-specific vector store with explicit persona_id and language
        vector_store = VectorStore(persona_settings, persona_id, language=language)

        # Cache the instance
        self.active_vector_stores[persona_id] = vector_store

        self.logger.info(f"Initialized vector store for persona: {persona_id} (language: {language})")
        return vector_store
    
    
    def _create_persona_settings(self, persona_id: str) -> Settings:
        """
        Create persona-specific settings
        
        Args:
            persona_id: The persona identifier
            
        Returns:
            Settings object with persona-specific paths
        """
        # Clone the base settings
        import copy
        persona_settings = copy.deepcopy(self.settings)
        
        # Update paths to be persona-specific
        persona_dir = self.personas_base_dir / persona_id
        
        # Update vector DB path
        persona_settings.vector_db.config['persist_directory'] = str(persona_dir / "vector_db")
        
        # Update collection name to be persona-specific
        persona_settings.vector_db.config['collection_name'] = f"{persona_id}_documents"
        
        return persona_settings
    
    def list_personas(self) -> List[Dict[str, Any]]:
        """
        List all registered personas
        
        Returns:
            List of persona information
        """
        return list(self.persona_registry.values())
    
    def get_persona_info(self, persona_id: str) -> Dict[str, Any]:
        """
        Get information about a specific persona
        
        Args:
            persona_id: The persona identifier
            
        Returns:
            Persona information dictionary
        """
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_id}' not found")
        
        return self.persona_registry[persona_id]
    
    def update_persona_stats(self, persona_id: str, stats: Dict[str, Any]):
        """
        Update statistics for a persona
        
        Args:
            persona_id: The persona identifier
            stats: Statistics to update
        """
        if persona_id not in self.persona_registry:
            return
        
        self.persona_registry[persona_id]["stats"].update(stats)
        self.persona_registry[persona_id]["stats"]["stats_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.persona_registry[persona_id]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save_registry()
    
    def delete_persona(self, persona_id: str, delete_data: bool = False):
        """
        Delete a persona from the registry
        
        Args:
            persona_id: The persona identifier
            delete_data: If True, also delete all data files
        """
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_id}' not found")
        
        # Close active connections
        if persona_id in self.active_vector_stores:
            self.active_vector_stores[persona_id].close()
            del self.active_vector_stores[persona_id]
        
        
        # Delete from registry
        del self.persona_registry[persona_id]
        self._save_registry()
        
        # Optionally delete data files
        if delete_data:
            import shutil
            persona_dir = self.personas_base_dir / persona_id
            if persona_dir.exists():
                shutil.rmtree(persona_dir)
                self.logger.info(f"Deleted all data for persona: {persona_id}")
        
        self.logger.info(f"Deleted persona from registry: {persona_id}")
    
    def cleanup(self):
        """Clean up resources"""
        # Close all active vector stores
        for persona_id, vector_store in self.active_vector_stores.items():
            try:
                vector_store.close()
            except Exception as e:
                self.logger.warning(f"Error closing vector store for {persona_id}: {e}")
        
        self.active_vector_stores.clear()
        self.logger.info("Persona manager cleanup complete")
    
    def persona_exists(self, persona_id: str) -> bool:
        """
        Check if a persona exists
        
        Args:
            persona_id: The persona identifier
            
        Returns:
            True if persona exists, False otherwise
        """
        return persona_id in self.persona_registry
    
    def get_or_create_persona(self, persona_name: str, metadata: Optional[Dict[str, Any]] = None, language: str = "en") -> str:
        """
        Get existing persona ID or create new one

        Args:
            persona_name: Human-readable persona name
            metadata: Optional metadata about the persona
            language: Content language code ('en' for English, 'zh' for Chinese)

        Returns:
            The persona ID
        """
        persona_id = self._sanitize_persona_id(persona_name)

        if not self.persona_exists(persona_id):
            # Add language to metadata
            full_metadata = metadata or {}
            full_metadata['language'] = language
            return self.register_persona(persona_name, full_metadata)

        return persona_id

    def get_persona_language(self, persona_id: str) -> str:
        """
        Get the language for a persona.

        Args:
            persona_id: Persona identifier

        Returns:
            Language code ('en', 'zh', etc.), defaults to 'en' if not set
        """
        if persona_id in self.persona_registry:
            metadata = self.persona_registry[persona_id].get('metadata', {})
            return metadata.get('language', 'en')
        return 'en'

    # Artifact Management Methods
    def _save_json(self, data: Union[Dict, List], file_path: Path, compress: bool = False):
        """Save data as JSON with optional compression."""
        try:
            if compress:
                with gzip.open(str(file_path) + ".gz", 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")
            raise
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data with optional decompression."""
        try:
            # Try compressed version first
            if Path(str(file_path) + ".gz").exists():
                with gzip.open(str(file_path) + ".gz", 'rt', encoding='utf-8') as f:
                    return json.load(f)
            # Fall back to uncompressed
            elif file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"No JSON file found at {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            raise
    
    def save_persona_constitution(self, persona: PersonaConstitution, persona_name: str, 
                                 compress: bool = True, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a persona constitution.
        
        Args:
            persona: PersonaConstitution object to save
            persona_name: Name identifier for the persona
            compress: Whether to compress the saved file
            metadata: Optional metadata about the artifact
            
        Returns:
            Artifact ID of the saved persona
        """
        persona_id = self._sanitize_persona_id(persona_name)
        
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_name}' not registered. Register it first.")
        
        # Create persona directory and artifacts subdirectory
        persona_dir = self.personas_base_dir / persona_id
        artifacts_dir = persona_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Generate artifact ID and filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_id = f"persona_{persona_id}_{timestamp}"
        filename = f"{artifact_id}.json"
        
        # Save the persona constitution
        file_path = artifacts_dir / filename
        persona_dict = persona.model_dump(mode='json')
        self._save_json(persona_dict, file_path, compress=compress)
        
        # Create artifact entry
        artifact_entry = {
            "artifact_id": artifact_id,
            "filename": filename + (".gz" if compress else ""),
            "created_at": datetime.utcnow().isoformat(),
            "compressed": compress,
            "file_path": str(file_path) + (".gz" if compress else ""),
            "quality_score": persona.overall_quality_score,
            "completeness_score": persona.completeness_score,
            "metadata": metadata
        }
        
        # Add artifact to persona registry
        self.persona_registry[persona_id]["artifacts"].append(artifact_entry)
        self.persona_registry[persona_id]["latest_artifact"] = artifact_id
        self.persona_registry[persona_id]["last_updated"] = datetime.utcnow().isoformat()
        
        self._save_registry()
        
        self.logger.info(f"Saved persona constitution for '{persona_name}' as {artifact_id}")
        return artifact_id
    
    def load_persona_constitution(self, persona_name: str, artifact_id: Optional[str] = None) -> PersonaConstitution:
        """
        Load a persona constitution.
        
        Args:
            persona_name: Name identifier for the persona
            artifact_id: Specific artifact ID to load (loads latest if None)
            
        Returns:
            PersonaConstitution object
        """
        persona_id = self._sanitize_persona_id(persona_name)
        
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_name}' not found in registry")
        
        persona_info = self.persona_registry[persona_id]
        
        # Find the artifact to load
        if artifact_id is None:
            # Load latest artifact
            if not persona_info.get("artifacts"):
                raise ValueError(f"No artifacts found for persona '{persona_name}'")
            artifact_entry = persona_info["artifacts"][-1]  # Latest artifact
        else:
            # Find specific artifact
            artifact_entry = None
            for entry in persona_info["artifacts"]:
                if entry["artifact_id"] == artifact_id:
                    artifact_entry = entry
                    break
            
            if artifact_entry is None:
                raise ValueError(f"Artifact '{artifact_id}' not found for persona '{persona_name}'")
        
        # Load the artifact
        artifact_path = Path(artifact_entry["file_path"])
        if artifact_entry["compressed"]:
            # Remove .gz suffix for _load_json to handle it
            artifact_path = Path(str(artifact_path).replace('.gz', ''))
        
        persona_data = self._load_json(artifact_path)
        return PersonaConstitution(**persona_data)
    
    def export_persona(self, persona_name: str, output_path: Path, format: str = "json") -> Path:
        """
        Export a persona constitution to a file.
        
        Args:
            persona_name: Name identifier for the persona
            output_path: Path to save the exported file
            format: Export format (currently only 'json' supported)
            
        Returns:
            Path to the exported file
        """
        if format != "json":
            raise ValueError(f"Unsupported export format: {format}")
        
        persona = self.load_persona_constitution(persona_name)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as uncompressed JSON for better readability
        self._save_json(persona.model_dump(mode='json'), output_path, compress=False)
        
        self.logger.info(f"Exported persona '{persona_name}' to {output_path}")
        return output_path
    
    def get_persona_stats(self, persona_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a persona.
        
        Args:
            persona_name: Name identifier for the persona
            
        Returns:
            Dictionary containing persona statistics
        """
        persona_id = self._sanitize_persona_id(persona_name)
        
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_name}' not found")
        
        persona_info = self.persona_registry[persona_id]
        
        try:
            persona = self.load_persona_constitution(persona_name)
            return {
                "name": persona_info["name"],
                "id": persona_info["id"],
                "created_at": persona_info.get("created_at"),
                "last_updated": persona_info.get("last_updated"),
                "artifacts_count": len(persona_info.get("artifacts", [])),
                "latest_artifact": persona_info.get("latest_artifact"),
                "stats": persona_info.get("stats", {}),
                "metadata": persona_info.get("metadata", {}),
                "quality_summary": persona.get_quality_summary()
            }
        except Exception as e:
            self.logger.error(f"Failed to load persona constitution for stats: {e}")
            return {
                "name": persona_info["name"],
                "id": persona_info["id"],
                "created_at": persona_info.get("created_at"),
                "last_updated": persona_info.get("last_updated"),
                "artifacts_count": len(persona_info.get("artifacts", [])),
                "latest_artifact": persona_info.get("latest_artifact"),
                "stats": persona_info.get("stats", {}),
                "metadata": persona_info.get("metadata", {}),
                "error": str(e)
            }