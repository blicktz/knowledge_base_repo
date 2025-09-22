"""
Persona Manager for multi-tenant architecture
Manages persona-specific vector stores and ensures data isolation
"""

import re
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
import json

from ..data.storage.vector_store import VectorStore
from ..data.storage.artifacts import ArtifactManager
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
        self.personas_base_dir = Path(self.settings.storage.artifacts_dir).parent / "personas"
        self.personas_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded vector stores and artifact managers
        self.active_vector_stores: Dict[str, VectorStore] = {}
        self.active_artifact_managers: Dict[str, ArtifactManager] = {}
        
        # Load persona registry
        self.registry_path = self.personas_base_dir / "persona_registry.json"
        self.persona_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the persona registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load persona registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the persona registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.persona_registry, f, indent=2)
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
        
        # Create persona entry
        self.persona_registry[persona_id] = {
            "name": persona_name,
            "id": persona_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "stats": {
                "documents": 0,
                "chunks": 0,
                "last_updated": None
            }
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
        
        # Create persona-specific settings
        persona_settings = self._create_persona_settings(persona_id)
        
        # Initialize persona-specific vector store
        vector_store = VectorStore(persona_settings)
        
        # Cache the instance
        self.active_vector_stores[persona_id] = vector_store
        
        self.logger.info(f"Initialized vector store for persona: {persona_id}")
        return vector_store
    
    def get_persona_artifact_manager(self, persona_id: str) -> ArtifactManager:
        """
        Get or create an artifact manager for a specific persona
        
        Args:
            persona_id: The persona identifier
            
        Returns:
            Persona-specific artifact manager instance
        """
        # Check if persona is registered
        if persona_id not in self.persona_registry:
            raise ValueError(f"Persona '{persona_id}' not registered. Register it first.")
        
        # Return cached instance if available
        if persona_id in self.active_artifact_managers:
            return self.active_artifact_managers[persona_id]
        
        # Create persona-specific settings
        persona_settings = self._create_persona_settings(persona_id)
        
        # Initialize persona-specific artifact manager
        artifact_manager = ArtifactManager(persona_settings)
        
        # Cache the instance
        self.active_artifact_managers[persona_id] = artifact_manager
        
        self.logger.info(f"Initialized artifact manager for persona: {persona_id}")
        return artifact_manager
    
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
        
        # Update artifacts path
        persona_settings.storage.artifacts_dir = str(persona_dir / "artifacts")
        
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
        self.persona_registry[persona_id]["stats"]["last_updated"] = datetime.now().isoformat()
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
        
        if persona_id in self.active_artifact_managers:
            del self.active_artifact_managers[persona_id]
        
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
        self.active_artifact_managers.clear()
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
    
    def get_or_create_persona(self, persona_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Get existing persona ID or create new one
        
        Args:
            persona_name: Human-readable persona name
            metadata: Optional metadata about the persona
            
        Returns:
            The persona ID
        """
        persona_id = self._sanitize_persona_id(persona_name)
        
        if not self.persona_exists(persona_id):
            return self.register_persona(persona_name, metadata)
        
        return persona_id