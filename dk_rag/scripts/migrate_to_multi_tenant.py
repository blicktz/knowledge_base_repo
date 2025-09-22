#!/usr/bin/env python
"""
Migration script to convert existing single-tenant data to multi-tenant architecture
"""

import argparse
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.data.storage.vector_store import VectorStore
from dk_rag.data.storage.artifacts import ArtifactManager
from dk_rag.utils.logging import setup_logger


class MultiTenantMigration:
    """
    Migrates existing single-tenant data to multi-tenant architecture
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("migration", level="INFO")
        self.persona_manager = PersonaManager(settings)
        
        # Backup paths
        self.backup_dir = Path(settings.storage.artifacts_dir).parent / "backup_pre_migration"
        
    def backup_existing_data(self):
        """Create backup of existing data before migration"""
        self.logger.info("Creating backup of existing data...")
        
        # Create backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.backup_dir / timestamp
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup vector DB
        vector_db_path = Path(self.settings.get_vector_db_path())
        if vector_db_path.exists():
            backup_vector_db = self.backup_dir / "vector_db"
            shutil.copytree(vector_db_path, backup_vector_db)
            self.logger.info(f"Backed up vector DB to: {backup_vector_db}")
        
        # Backup artifacts
        artifacts_path = Path(self.settings.get_artifacts_path())
        if artifacts_path.exists():
            backup_artifacts = self.backup_dir / "artifacts"
            shutil.copytree(artifacts_path, backup_artifacts)
            self.logger.info(f"Backed up artifacts to: {backup_artifacts}")
        
        # Save backup info
        backup_info = {
            "timestamp": timestamp,
            "vector_db_backed_up": vector_db_path.exists(),
            "artifacts_backed_up": artifacts_path.exists(),
            "backup_location": str(self.backup_dir)
        }
        
        with open(self.backup_dir / "backup_info.json", 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        self.logger.info(f"Backup complete: {self.backup_dir}")
        return self.backup_dir
    
    def migrate_artifacts(self, persona_name: str):
        """Migrate existing artifacts to persona-specific directory"""
        self.logger.info(f"Migrating artifacts for persona: {persona_name}")
        
        # Register persona
        persona_id = self.persona_manager.get_or_create_persona(persona_name)
        
        # Get old and new artifact paths
        old_artifacts_dir = Path(self.settings.get_artifacts_path())
        new_artifacts_dir = Path(self.settings.get_artifacts_path(persona_id))
        
        # Create new directory
        new_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and migrate persona-specific artifacts
        migrated_count = 0
        pattern = f"persona_{persona_id}_*.json*"
        
        for artifact_file in old_artifacts_dir.glob(pattern):
            if artifact_file.is_file() and not artifact_file.is_symlink():
                # Copy to new location
                new_path = new_artifacts_dir / artifact_file.name
                shutil.copy2(artifact_file, new_path)
                self.logger.info(f"Migrated artifact: {artifact_file.name}")
                migrated_count += 1
        
        # Also check for generic persona files
        alt_pattern = f"persona_{persona_name}_*.json*"
        for artifact_file in old_artifacts_dir.glob(alt_pattern):
            if artifact_file.is_file() and not artifact_file.is_symlink():
                # Check if already migrated
                new_path = new_artifacts_dir / artifact_file.name
                if not new_path.exists():
                    shutil.copy2(artifact_file, new_path)
                    self.logger.info(f"Migrated artifact: {artifact_file.name}")
                    migrated_count += 1
        
        self.logger.info(f"Migrated {migrated_count} artifacts for persona '{persona_name}'")
        return migrated_count
    
    def migrate_vector_store(self, persona_name: str, source_collection: str = "influencer_transcripts"):
        """Migrate vector store data to persona-specific collection"""
        self.logger.info(f"Migrating vector store for persona: {persona_name}")
        
        # Register persona
        persona_id = self.persona_manager.get_or_create_persona(persona_name)
        
        # Initialize old vector store
        old_vector_store = VectorStore(self.settings)
        
        # Check if source collection exists
        try:
            old_stats = old_vector_store.get_collection_stats()
            total_chunks = old_stats.get('total_chunks', 0)
            
            if total_chunks == 0:
                self.logger.warning(f"No data found in source collection: {source_collection}")
                return 0
            
            self.logger.info(f"Found {total_chunks} chunks in source collection")
        except Exception as e:
            self.logger.error(f"Error accessing source collection: {e}")
            return 0
        
        # Get persona-specific vector store
        new_vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        # Migrate data in batches
        batch_size = 100
        migrated_count = 0
        
        try:
            # Get all data from old collection
            # Note: This is a simplified approach. For large datasets,
            # you'd want to implement proper pagination
            all_data = old_vector_store.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if all_data and all_data.get('documents'):
                documents = all_data['documents']
                metadatas = all_data.get('metadatas', [{}] * len(documents))
                embeddings = all_data.get('embeddings', [])
                ids = all_data.get('ids', [])
                
                # Process in batches
                for i in range(0, len(documents), batch_size):
                    batch_end = min(i + batch_size, len(documents))
                    
                    # Update metadata with persona_id
                    batch_metadatas = []
                    for j in range(i, batch_end):
                        metadata = metadatas[j] if j < len(metadatas) else {}
                        metadata['persona_id'] = persona_id
                        metadata['migrated_from'] = source_collection
                        metadata['migration_date'] = datetime.now().isoformat()
                        batch_metadatas.append(metadata)
                    
                    # Add to new collection
                    if embeddings:
                        new_vector_store.collection.add(
                            documents=documents[i:batch_end],
                            metadatas=batch_metadatas,
                            embeddings=embeddings[i:batch_end],
                            ids=[f"{persona_id}_{id}" for id in ids[i:batch_end]]
                        )
                    else:
                        # Re-generate embeddings if not available
                        batch_docs = [
                            {'content': doc, 'persona_id': persona_id, **meta}
                            for doc, meta in zip(documents[i:batch_end], batch_metadatas)
                        ]
                        new_vector_store.add_documents(batch_docs)
                    
                    migrated_count += (batch_end - i)
                    self.logger.info(f"Migrated batch {i//batch_size + 1}: {batch_end - i} documents")
                
                self.logger.info(f"Successfully migrated {migrated_count} chunks to persona '{persona_name}'")
                
                # Update persona stats
                self.persona_manager.update_persona_stats(persona_id, {
                    'chunks': migrated_count,
                    'migration_date': datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            return migrated_count
        finally:
            # Close connections
            old_vector_store.close()
        
        return migrated_count
    
    def verify_migration(self, persona_name: str):
        """Verify that migration was successful"""
        self.logger.info(f"Verifying migration for persona: {persona_name}")
        
        persona_id = self.persona_manager._sanitize_persona_id(persona_name)
        
        # Check persona registration
        if not self.persona_manager.persona_exists(persona_id):
            self.logger.error(f"Persona '{persona_name}' not found in registry")
            return False
        
        # Check vector store
        try:
            vector_store = self.persona_manager.get_persona_vector_store(persona_id)
            stats = vector_store.get_collection_stats()
            chunk_count = stats.get('total_chunks', 0)
            self.logger.info(f"Vector store has {chunk_count} chunks")
        except Exception as e:
            self.logger.error(f"Error accessing vector store: {e}")
            return False
        
        # Check artifacts
        try:
            artifact_manager = self.persona_manager.get_persona_artifact_manager(persona_id)
            artifacts = artifact_manager.list_artifacts()
            self.logger.info(f"Found {len(artifacts)} artifacts")
        except Exception as e:
            self.logger.error(f"Error accessing artifacts: {e}")
            return False
        
        self.logger.info(f"Migration verified successfully for persona '{persona_name}'")
        return True
    
    def run_migration(self, persona_name: str, skip_backup: bool = False):
        """Run the complete migration process"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Multi-Tenant Migration")
        self.logger.info("=" * 60)
        
        # Step 1: Backup
        if not skip_backup:
            self.backup_existing_data()
        else:
            self.logger.warning("Skipping backup (not recommended)")
        
        # Step 2: Migrate artifacts
        self.logger.info("\nMigrating artifacts...")
        artifact_count = self.migrate_artifacts(persona_name)
        
        # Step 3: Migrate vector store
        self.logger.info("\nMigrating vector store...")
        chunk_count = self.migrate_vector_store(persona_name)
        
        # Step 4: Verify
        self.logger.info("\nVerifying migration...")
        success = self.verify_migration(persona_name)
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Migration Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Persona: {persona_name}")
        self.logger.info(f"Artifacts migrated: {artifact_count}")
        self.logger.info(f"Chunks migrated: {chunk_count}")
        self.logger.info(f"Verification: {'PASSED' if success else 'FAILED'}")
        
        if not skip_backup:
            self.logger.info(f"Backup location: {self.backup_dir}")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing data to multi-tenant architecture"
    )
    
    parser.add_argument(
        "--persona-name",
        required=True,
        help="Name of the persona to migrate (e.g., 'Dan Kennedy')"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file",
        default=None
    )
    
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup step (not recommended)"
    )
    
    parser.add_argument(
        "--source-collection",
        default="influencer_transcripts",
        help="Source collection name in vector store"
    )
    
    args = parser.parse_args()
    
    # Load settings
    if args.config:
        settings = Settings.from_file(args.config)
    else:
        settings = Settings.from_default_config()
    
    # Run migration
    migrator = MultiTenantMigration(settings)
    success = migrator.run_migration(
        persona_name=args.persona_name,
        skip_backup=args.skip_backup
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()