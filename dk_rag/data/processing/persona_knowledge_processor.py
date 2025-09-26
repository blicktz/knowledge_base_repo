"""
Persona Knowledge Processor

This module processes Phase 1 JSON artifacts to extract mental models
and core beliefs for indexing in their respective RAG systems.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from ...models.knowledge_types import KnowledgeType
from ...models.knowledge_results import IndexingResult
from ...utils.logging import get_logger
from ...utils.artifact_discovery import ArtifactDiscovery, ArtifactDiscoveryError


@dataclass
class ProcessingResult:
    """Result of processing a persona JSON file."""
    mental_models: List[Dict[str, Any]]
    core_beliefs: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    source_file: str
    processing_timestamp: datetime


class PersonaKnowledgeProcessor:
    """
    Processes Phase 1 persona JSON files to extract knowledge for RAG indexing.
    
    Handles robust JSON parsing, validation, and extraction of mental models
    and core beliefs with comprehensive error handling and logging.
    """
    
    def __init__(self):
        """Initialize the processor."""
        self.logger = get_logger(__name__)
        
        # Schema validation for expected fields
        self.mental_model_required_fields = {'name', 'description', 'steps', 'categories'}
        self.mental_model_optional_fields = {'confidence_score', 'frequency'}
        
        self.core_belief_required_fields = {'statement', 'category', 'supporting_evidence'}
        self.core_belief_optional_fields = {'confidence_score', 'frequency'}
    
    def process_persona_file(
        self,
        json_path: Union[str, Path],
        validate_schema: bool = True,
        knowledge_type: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a Phase 1 persona JSON file.
        
        Args:
            json_path: Path to the persona JSON file
            validate_schema: Whether to validate the JSON schema
            knowledge_type: Type of knowledge to extract ('mental_models', 'core_beliefs', or None for both)
            
        Returns:
            ProcessingResult with extracted data and any errors/warnings
        """
        json_path = Path(json_path)
        self.logger.info(f"Processing persona file: {json_path}")
        
        result = ProcessingResult(
            mental_models=[],
            core_beliefs=[],
            errors=[],
            warnings=[],
            source_file=str(json_path),
            processing_timestamp=datetime.now()
        )
        
        try:
            # Load and parse JSON
            persona_data = self._load_json_file(json_path)
            
            # Extract mental models (only if requested)
            if knowledge_type is None or knowledge_type == 'mental_models':
                mental_models = self._extract_mental_models(
                    persona_data, 
                    validate_schema=validate_schema
                )
                result.mental_models = mental_models['data']
                result.errors.extend(mental_models['errors'])
                result.warnings.extend(mental_models['warnings'])
            
            # Extract core beliefs (only if requested)
            if knowledge_type is None or knowledge_type == 'core_beliefs':
                core_beliefs = self._extract_core_beliefs(
                    persona_data,
                    validate_schema=validate_schema
                )
                result.core_beliefs = core_beliefs['data']
                result.errors.extend(core_beliefs['errors'])
                result.warnings.extend(core_beliefs['warnings'])
            
            self.logger.info(
                f"Extracted {len(result.mental_models)} mental models and "
                f"{len(result.core_beliefs)} core beliefs from {json_path.name}"
            )
            
            if result.errors:
                self.logger.warning(f"Processing completed with {len(result.errors)} errors")
            
        except Exception as e:
            error_msg = f"Critical error processing {json_path}: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _load_json_file(self, json_path: Path) -> Dict[str, Any]:
        """
        Load and parse JSON file with error handling.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Parsed JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Persona file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.debug(f"Successfully loaded JSON from {json_path}")
            return data
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Malformed JSON in {json_path}: {e.msg}",
                e.doc,
                e.pos
            )
        except Exception as e:
            raise Exception(f"Failed to read {json_path}: {e}")
    
    def _extract_mental_models(
        self,
        persona_data: Dict[str, Any],
        validate_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Extract mental models from persona data.
        
        Args:
            persona_data: Parsed persona JSON data
            validate_schema: Whether to validate each mental model
            
        Returns:
            Dictionary with extracted data, errors, and warnings
        """
        result = {'data': [], 'errors': [], 'warnings': []}
        
        try:
            mental_models_raw = persona_data.get('mental_models', [])
            
            if not mental_models_raw:
                result['warnings'].append("No mental_models section found in persona data")
                return result
            
            if not isinstance(mental_models_raw, list):
                result['errors'].append("mental_models section is not a list")
                return result
            
            self.logger.debug(f"Found {len(mental_models_raw)} mental models to process")
            
            for i, model_data in enumerate(mental_models_raw):
                try:
                    # Validate schema if requested
                    if validate_schema:
                        validation_errors = self._validate_mental_model_schema(model_data, i)
                        if validation_errors:
                            result['warnings'].extend(validation_errors)
                            continue
                    
                    # Process and clean the mental model
                    processed_model = self._process_mental_model(model_data, i)
                    result['data'].append(processed_model)
                    
                except Exception as e:
                    error_msg = f"Error processing mental model {i}: {e}"
                    self.logger.error(error_msg)
                    result['errors'].append(error_msg)
                    continue
            
            self.logger.info(f"Successfully processed {len(result['data'])} mental models")
            
        except Exception as e:
            error_msg = f"Critical error extracting mental models: {e}"
            result['errors'].append(error_msg)
        
        return result
    
    def _extract_core_beliefs(
        self,
        persona_data: Dict[str, Any],
        validate_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Extract core beliefs from persona data.
        
        Args:
            persona_data: Parsed persona JSON data
            validate_schema: Whether to validate each core belief
            
        Returns:
            Dictionary with extracted data, errors, and warnings
        """
        result = {'data': [], 'errors': [], 'warnings': []}
        
        try:
            core_beliefs_raw = persona_data.get('core_beliefs', [])
            
            if not core_beliefs_raw:
                result['warnings'].append("No core_beliefs section found in persona data")
                return result
            
            if not isinstance(core_beliefs_raw, list):
                result['errors'].append("core_beliefs section is not a list")
                return result
            
            self.logger.debug(f"Found {len(core_beliefs_raw)} core beliefs to process")
            
            for i, belief_data in enumerate(core_beliefs_raw):
                try:
                    # Validate schema if requested
                    if validate_schema:
                        validation_errors = self._validate_core_belief_schema(belief_data, i)
                        if validation_errors:
                            result['warnings'].extend(validation_errors)
                            continue
                    
                    # Process and clean the core belief
                    processed_belief = self._process_core_belief(belief_data, i)
                    result['data'].append(processed_belief)
                    
                except Exception as e:
                    error_msg = f"Error processing core belief {i}: {e}"
                    self.logger.error(error_msg)
                    result['errors'].append(error_msg)
                    continue
            
            self.logger.info(f"Successfully processed {len(result['data'])} core beliefs")
            
        except Exception as e:
            error_msg = f"Critical error extracting core beliefs: {e}"
            result['errors'].append(error_msg)
        
        return result
    
    def _validate_mental_model_schema(
        self, 
        model_data: Dict[str, Any], 
        index: int
    ) -> List[str]:
        """
        Validate mental model schema.
        
        Args:
            model_data: Mental model data to validate
            index: Index of the mental model for error reporting
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not isinstance(model_data, dict):
            errors.append(f"Mental model {index} is not a dictionary")
            return errors
        
        # Check required fields
        for field in self.mental_model_required_fields:
            if field not in model_data:
                errors.append(f"Mental model {index} missing required field: {field} (will be skipped)")
            elif not model_data[field]:
                errors.append(f"Mental model {index} has empty required field: {field} (will be skipped)")
        
        # Validate field types
        if 'name' in model_data and not isinstance(model_data['name'], str):
            errors.append(f"Mental model {index} 'name' must be a string")
        
        if 'description' in model_data and not isinstance(model_data['description'], str):
            errors.append(f"Mental model {index} 'description' must be a string")
        
        if 'steps' in model_data and not isinstance(model_data['steps'], list):
            errors.append(f"Mental model {index} 'steps' must be a list")
        
        if 'categories' in model_data and not isinstance(model_data['categories'], list):
            errors.append(f"Mental model {index} 'categories' must be a list")
        
        return errors
    
    def _validate_core_belief_schema(
        self, 
        belief_data: Dict[str, Any], 
        index: int
    ) -> List[str]:
        """
        Validate core belief schema.
        
        Args:
            belief_data: Core belief data to validate
            index: Index of the core belief for error reporting
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not isinstance(belief_data, dict):
            errors.append(f"Core belief {index} is not a dictionary")
            return errors
        
        # Check required fields
        for field in self.core_belief_required_fields:
            if field not in belief_data:
                errors.append(f"Core belief {index} missing required field: {field} (will be skipped)")
            elif not belief_data[field]:
                errors.append(f"Core belief {index} has empty required field: {field} (will be skipped)")
        
        # Validate field types
        if 'statement' in belief_data and not isinstance(belief_data['statement'], str):
            errors.append(f"Core belief {index} 'statement' must be a string")
        
        if 'category' in belief_data and not isinstance(belief_data['category'], str):
            errors.append(f"Core belief {index} 'category' must be a string")
        
        if 'supporting_evidence' in belief_data and not isinstance(belief_data['supporting_evidence'], list):
            errors.append(f"Core belief {index} 'supporting_evidence' must be a list")
        
        return errors
    
    def _process_mental_model(self, model_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Process and clean a mental model.
        
        Args:
            model_data: Raw mental model data
            index: Index for error reporting
            
        Returns:
            Processed mental model data
        """
        processed = {
            'name': str(model_data.get('name', '')).strip(),
            'description': str(model_data.get('description', '')).strip(),
            'steps': self._clean_list_field(model_data.get('steps', [])),
            'categories': self._clean_list_field(model_data.get('categories', [])),
            'confidence_score': float(model_data.get('confidence_score', 0.0)),
            'frequency': int(model_data.get('frequency', 0)),
            'index': index,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Generate content hash for deduplication
        content_for_hash = f"{processed['name']}{processed['description']}"
        processed['content_hash'] = hashlib.sha256(
            content_for_hash.encode('utf-8')
        ).hexdigest()[:16]
        
        return processed
    
    def _process_core_belief(self, belief_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Process and clean a core belief.
        
        Args:
            belief_data: Raw core belief data
            index: Index for error reporting
            
        Returns:
            Processed core belief data
        """
        processed = {
            'statement': str(belief_data.get('statement', '')).strip(),
            'category': str(belief_data.get('category', '')).strip(),
            'supporting_evidence': self._clean_list_field(
                belief_data.get('supporting_evidence', [])
            ),
            'confidence_score': float(belief_data.get('confidence_score', 0.0)),
            'frequency': int(belief_data.get('frequency', 0)),
            'index': index,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Generate content hash for deduplication
        content_for_hash = f"{processed['statement']}{processed['category']}"
        processed['content_hash'] = hashlib.sha256(
            content_for_hash.encode('utf-8')
        ).hexdigest()[:16]
        
        return processed
    
    def _clean_list_field(self, field_data: Any) -> List[str]:
        """
        Clean and validate list fields.
        
        Args:
            field_data: Field data that should be a list
            
        Returns:
            Cleaned list of strings
        """
        if not isinstance(field_data, list):
            return []
        
        cleaned = []
        for item in field_data:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif item is not None:
                # Convert non-string items to strings
                str_item = str(item).strip()
                if str_item:
                    cleaned.append(str_item)
        
        return cleaned
    
    def get_processing_statistics(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Get statistics about the processing operation.
        
        Args:
            result: ProcessingResult to analyze
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'source_file': result.source_file,
            'processing_timestamp': result.processing_timestamp.isoformat(),
            'mental_models': {
                'count': len(result.mental_models),
                'avg_confidence': 0.0,
                'categories': set(),
                'avg_steps': 0.0
            },
            'core_beliefs': {
                'count': len(result.core_beliefs),
                'avg_confidence': 0.0,
                'categories': set(),
                'avg_evidence': 0.0
            },
            'errors': len(result.errors),
            'warnings': len(result.warnings),
            'success': len(result.errors) == 0
        }
        
        # Mental models statistics
        if result.mental_models:
            confidences = [m.get('confidence_score', 0.0) for m in result.mental_models]
            stats['mental_models']['avg_confidence'] = sum(confidences) / len(confidences)
            
            for model in result.mental_models:
                stats['mental_models']['categories'].update(model.get('categories', []))
                
            steps_counts = [len(m.get('steps', [])) for m in result.mental_models]
            stats['mental_models']['avg_steps'] = sum(steps_counts) / len(steps_counts)
        
        stats['mental_models']['categories'] = list(stats['mental_models']['categories'])
        
        # Core beliefs statistics
        if result.core_beliefs:
            confidences = [b.get('confidence_score', 0.0) for b in result.core_beliefs]
            stats['core_beliefs']['avg_confidence'] = sum(confidences) / len(confidences)
            
            categories = [b.get('category', '') for b in result.core_beliefs]
            stats['core_beliefs']['categories'] = list(set(categories))
            
            evidence_counts = [len(b.get('supporting_evidence', [])) for b in result.core_beliefs]
            stats['core_beliefs']['avg_evidence'] = sum(evidence_counts) / len(evidence_counts)
        
        return stats
    
    def process_latest_artifact(
        self, 
        persona_id: str,
        settings: Optional[Any] = None,
        validate_schema: bool = True,
        knowledge_type: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process the latest Phase 1 artifact for a persona.
        
        Uses automatic artifact discovery to find and process the most recent
        Phase 1 JSON artifact for the given persona.
        
        Args:
            persona_id: Persona identifier
            settings: Settings instance for configuration
            validate_schema: Whether to validate the JSON schema
            knowledge_type: Type of knowledge to extract ('mental_models', 'core_beliefs', or None for both)
            
        Returns:
            ProcessingResult with extracted data and any errors/warnings
            
        Raises:
            ArtifactDiscoveryError: If artifact discovery fails
        """
        self.logger.info(f"Auto-processing latest artifact for persona: {persona_id}")
        
        try:
            # Initialize artifact discovery
            artifact_discovery = ArtifactDiscovery(settings)
            
            # Get latest artifact JSON
            json_path, artifact_info = artifact_discovery.get_latest_artifact_json(persona_id)
            
            self.logger.info(
                f"Processing artifact: {artifact_info.file_path.name} "
                f"(timestamp: {artifact_info.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
            )
            
            try:
                # Process the artifact
                result = self.process_persona_file(json_path, validate_schema, knowledge_type)
                
                # Add artifact metadata to result
                result.source_file = str(artifact_info.file_path)
                
                # Add processing note
                result.warnings.insert(0, 
                    f"Auto-processed artifact: {artifact_info.file_path.name} "
                    f"(extracted to: {json_path})"
                )
                
                self.logger.info(
                    f"Successfully processed artifact: "
                    f"{len(result.mental_models)} mental models, "
                    f"{len(result.core_beliefs)} core beliefs"
                )
                
                return result
                
            finally:
                # Clean up temporary file if it was created
                if artifact_info.is_compressed:
                    artifact_discovery.cleanup_temp_file(json_path)
                    
        except ArtifactDiscoveryError as e:
            # Create error result
            error_result = ProcessingResult(
                mental_models=[],
                core_beliefs=[],
                errors=[f"Artifact discovery failed: {e}"],
                warnings=[],
                source_file=f"persona_id:{persona_id}",
                processing_timestamp=datetime.now()
            )
            return error_result
            
        except Exception as e:
            self.logger.error(f"Failed to auto-process artifact for {persona_id}: {e}")
            
            # Create error result
            error_result = ProcessingResult(
                mental_models=[],
                core_beliefs=[],
                errors=[f"Auto-processing failed: {e}"],
                warnings=[],
                source_file=f"persona_id:{persona_id}",
                processing_timestamp=datetime.now()
            )
            return error_result