"""
Component Registry for Long-Lived RAG Components

This module provides a registry for caching expensive-to-create components
like PersonaManager and KnowledgeIndexer, enabling efficient resource usage
in server environments while maintaining CLI compatibility.
"""

import threading
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

from .logging import get_logger, get_component_logger


class ComponentRegistry:
    """
    Thread-safe registry for caching long-lived RAG components.
    
    Provides efficient component reuse for server deployments while
    maintaining backward compatibility for CLI operations.
    """
    
    def __init__(self):
        self.logger = get_component_logger("CompReg")
        self._lock = threading.RLock()
        
        # Component caches
        self._persona_managers: Dict[str, Any] = {}  # settings_hash -> PersonaManager
        self._knowledge_indexers: Dict[Tuple[str, str], Any] = {}  # (settings_hash, persona_id) -> KnowledgeIndexer
        
        # Component metadata
        self._component_info: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("🏗️  ComponentRegistry initialized")
    
    def _get_settings_hash(self, settings) -> str:
        """
        Create a hash key for settings to enable caching.
        
        Args:
            settings: Settings object
            
        Returns:
            String hash of relevant settings
        """
        # Create a simple hash based on key settings that affect component behavior
        try:
            # Get key paths that affect component behavior
            storage_dir = str(getattr(settings.storage, 'base_storage_dir', 'default'))
            vector_config = str(getattr(settings.vector_db, 'config', {}))
            agent_config = str(getattr(settings.agent, 'enabled', False))
            
            # Create a simple hash
            components = [storage_dir, vector_config, agent_config]
            settings_key = hash(tuple(components))
            return str(settings_key)
            
        except Exception as e:
            self.logger.warning(f"Failed to create settings hash: {e}")
            # Fallback to timestamp-based key (no caching)
            return str(datetime.now().timestamp())
    
    def get_persona_manager(self, settings):
        """
        Get or create a PersonaManager instance.
        
        Args:
            settings: Application settings
            
        Returns:
            PersonaManager instance
        """
        settings_hash = self._get_settings_hash(settings)
        
        with self._lock:
            # Return cached instance if available
            if settings_hash in self._persona_managers:
                self.logger.debug(f"Using cached PersonaManager for settings hash: {settings_hash}")
                return self._persona_managers[settings_hash]
            
            # Create new instance
            self.logger.info(f"🚀 Creating new PersonaManager for settings hash: {settings_hash}")
            try:
                from ..core.persona_manager import PersonaManager
                persona_manager = PersonaManager(settings)
                
                # Cache the instance
                self._persona_managers[settings_hash] = persona_manager
                
                # Store metadata
                self._component_info[f"persona_manager_{settings_hash}"] = {
                    "type": "PersonaManager",
                    "settings_hash": settings_hash,
                    "created_at": datetime.now().isoformat(),
                    "base_storage_dir": str(getattr(settings.storage, 'base_storage_dir', 'default'))
                }
                
                self.logger.info(f"✅ PersonaManager created and cached")
                return persona_manager
                
            except Exception as e:
                self.logger.error(f"Failed to create PersonaManager: {e}")
                raise
    
    def get_knowledge_indexer(self, settings, persona_id: str):
        """
        Get or create a KnowledgeIndexer instance.
        
        Args:
            settings: Application settings
            persona_id: Persona identifier
            
        Returns:
            KnowledgeIndexer instance
        """
        settings_hash = self._get_settings_hash(settings)
        cache_key = (settings_hash, persona_id)
        
        with self._lock:
            # Return cached instance if available
            if cache_key in self._knowledge_indexers:
                self.logger.debug(f"Using cached KnowledgeIndexer for persona: {persona_id}")
                return self._knowledge_indexers[cache_key]
            
            # Create new instance
            self.logger.info(f"🚀 Creating new KnowledgeIndexer for persona: {persona_id}")
            try:
                from ..core.knowledge_indexer import KnowledgeIndexer
                
                # Get or create PersonaManager first
                persona_manager = self.get_persona_manager(settings)
                
                # Create KnowledgeIndexer
                knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
                
                # Cache the instance
                self._knowledge_indexers[cache_key] = knowledge_indexer
                
                # Store metadata
                self._component_info[f"knowledge_indexer_{settings_hash}_{persona_id}"] = {
                    "type": "KnowledgeIndexer",
                    "settings_hash": settings_hash,
                    "persona_id": persona_id,
                    "created_at": datetime.now().isoformat()
                }
                
                self.logger.info(f"✅ KnowledgeIndexer created and cached for persona: {persona_id}")
                return knowledge_indexer
                
            except Exception as e:
                self.logger.error(f"Failed to create KnowledgeIndexer for persona {persona_id}: {e}")
                raise
    
    def is_component_cached(self, component_type: str, settings, persona_id: Optional[str] = None) -> bool:
        """
        Check if a component is already cached.
        
        Args:
            component_type: Type of component ('persona_manager' or 'knowledge_indexer')
            settings: Application settings
            persona_id: Persona ID (required for knowledge_indexer)
            
        Returns:
            True if component is cached, False otherwise
        """
        settings_hash = self._get_settings_hash(settings)
        
        with self._lock:
            if component_type == "persona_manager":
                return settings_hash in self._persona_managers
            elif component_type == "knowledge_indexer":
                if persona_id is None:
                    return False
                cache_key = (settings_hash, persona_id)
                return cache_key in self._knowledge_indexers
            else:
                return False
    
    def clear_components(self, component_type: Optional[str] = None):
        """
        Clear cached components.
        
        Args:
            component_type: Type to clear ('persona_manager', 'knowledge_indexer'), or None for all
        """
        with self._lock:
            if component_type is None or component_type == "persona_manager":
                cleared_count = len(self._persona_managers)
                self._persona_managers.clear()
                self.logger.info(f"🧹 Cleared {cleared_count} cached PersonaManager instances")
            
            if component_type is None or component_type == "knowledge_indexer":
                cleared_count = len(self._knowledge_indexers)
                self._knowledge_indexers.clear()
                self.logger.info(f"🧹 Cleared {cleared_count} cached KnowledgeIndexer instances")
            
            # Clear corresponding metadata
            keys_to_remove = []
            for key, info in self._component_info.items():
                if component_type is None or info.get("type", "").lower().replace("manager", "_manager") == component_type:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._component_info[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            stats = {
                "total_components_cached": len(self._component_info),
                "persona_managers_cached": len(self._persona_managers),
                "knowledge_indexers_cached": len(self._knowledge_indexers),
                "component_details": self._component_info.copy()
            }
            
            return stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get approximate memory usage information.
        
        Returns:
            Dictionary with memory usage estimates
        """
        with self._lock:
            usage = {
                "cached_persona_managers": len(self._persona_managers),
                "cached_knowledge_indexers": len(self._knowledge_indexers),
                "total_cached_components": len(self._component_info),
                "estimated_memory_savings": "High (shared models and persistent connections)"
            }
            
            return usage


# Global component registry instance
_component_registry = None


def get_component_registry() -> ComponentRegistry:
    """Get the global component registry instance"""
    global _component_registry
    if _component_registry is None:
        _component_registry = ComponentRegistry()
    return _component_registry