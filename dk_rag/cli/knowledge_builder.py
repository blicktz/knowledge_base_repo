"""
Knowledge Builder CLI

Command-line interface for building and managing mental models and 
core beliefs knowledge bases from Phase 1 persona JSON artifacts.
"""

import click
import json
from typing import Optional
from pathlib import Path
from datetime import datetime

from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..config.settings import Settings
from ..models.knowledge_types import KnowledgeType
from ..utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
@click.pass_context
def knowledge(ctx):
    """Knowledge base management commands for mental models and core beliefs."""
    # Initialize shared components
    ctx.ensure_object(dict)
    ctx.obj['settings'] = Settings.from_default_config()
    ctx.obj['persona_manager'] = PersonaManager(ctx.obj['settings'])


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--json-path', required=True, help='Path to Phase 1 persona JSON file')
@click.option('--rebuild', is_flag=True, help='Rebuild existing index from scratch')
@click.option('--validate/--no-validate', default=True, help='Validate JSON schema (default: validate)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def build_mental_models(ctx, persona_id: str, json_path: str, rebuild: bool, validate: bool, verbose: bool):
    """Build mental models knowledge base from Phase 1 JSON."""
    
    if verbose:
        click.echo(f"Building mental models index for persona: {persona_id}")
        click.echo(f"Source JSON: {json_path}")
        click.echo(f"Rebuild: {rebuild}")
        click.echo(f"Validate: {validate}")
    
    try:
        # Validate inputs
        json_file = Path(json_path)
        if not json_file.exists():
            click.echo(f"❌ Error: JSON file not found: {json_path}", err=True)
            return
        
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Build index
        with click.progressbar(length=100, label='Building mental models index') as bar:
            bar.update(10)  # Start
            
            result = indexer.build_mental_models_index(
                persona_id=persona_id,
                json_path=json_path,
                rebuild=rebuild,
                validate=validate
            )
            
            bar.update(100)  # Complete
        
        # Display results
        if result.success:
            click.echo(f"✅ Mental models index built successfully!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings and verbose:
                click.echo("Warnings:")
                for warning in result.warnings:
                    click.echo(f"   - {warning}")
        elif result.partial_success:
            click.echo(f"⚠️ Mental models index built with warnings!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings:
                click.echo("Warnings:")
                for warning in result.warnings[:10]:  # Show first 10 warnings
                    click.echo(f"   - {warning}")
                if len(result.warnings) > 10:
                    click.echo(f"   ... and {len(result.warnings) - 10} more warnings")
        else:
            click.echo(f"❌ Failed to build mental models index")
            if result.errors:
                click.echo("Errors:")
                for error in result.errors:
                    click.echo(f"   - {error}")
    
    except Exception as e:
        click.echo(f"❌ Critical error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--json-path', required=True, help='Path to Phase 1 persona JSON file')
@click.option('--rebuild', is_flag=True, help='Rebuild existing index from scratch')
@click.option('--validate/--no-validate', default=True, help='Validate JSON schema (default: validate)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def build_core_beliefs(ctx, persona_id: str, json_path: str, rebuild: bool, validate: bool, verbose: bool):
    """Build core beliefs knowledge base from Phase 1 JSON."""
    
    if verbose:
        click.echo(f"Building core beliefs index for persona: {persona_id}")
        click.echo(f"Source JSON: {json_path}")
        click.echo(f"Rebuild: {rebuild}")
        click.echo(f"Validate: {validate}")
    
    try:
        # Validate inputs
        json_file = Path(json_path)
        if not json_file.exists():
            click.echo(f"❌ Error: JSON file not found: {json_path}", err=True)
            return
        
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Build index
        with click.progressbar(length=100, label='Building core beliefs index') as bar:
            bar.update(10)  # Start
            
            result = indexer.build_core_beliefs_index(
                persona_id=persona_id,
                json_path=json_path,
                rebuild=rebuild,
                validate=validate
            )
            
            bar.update(100)  # Complete
        
        # Display results
        if result.success:
            click.echo(f"✅ Core beliefs index built successfully!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings and verbose:
                click.echo("Warnings:")
                for warning in result.warnings:
                    click.echo(f"   - {warning}")
        elif result.partial_success:
            click.echo(f"⚠️ Core beliefs index built with warnings!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings:
                click.echo("Warnings:")
                for warning in result.warnings[:10]:  # Show first 10 warnings
                    click.echo(f"   - {warning}")
                if len(result.warnings) > 10:
                    click.echo(f"   ... and {len(result.warnings) - 10} more warnings")
        else:
            click.echo(f"❌ Failed to build core beliefs index")
            if result.errors:
                click.echo("Errors:")
                for error in result.errors:
                    click.echo(f"   - {error}")
    
    except Exception as e:
        click.echo(f"❌ Critical error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--query', required=True, help='Search query')
@click.option('--top-k', default=5, help='Number of results to return (default: 5)')
@click.option('--min-confidence', default=0.0, help='Minimum confidence score (default: 0.0)')
@click.option('--categories', help='Filter by categories (comma-separated)')
@click.option('--no-rerank', is_flag=True, help='Disable reranking')
@click.option('--show-scores', is_flag=True, help='Show relevance scores')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def search_mental_models(ctx, persona_id: str, query: str, top_k: int, min_confidence: float, 
                        categories: Optional[str], no_rerank: bool, show_scores: bool, 
                        output_format: str, verbose: bool):
    """Search mental models knowledge base."""
    
    if verbose:
        click.echo(f"Searching mental models for persona: {persona_id}")
        click.echo(f"Query: {query}")
        click.echo(f"Top-K: {top_k}")
        click.echo(f"Min confidence: {min_confidence}")
        if categories:
            click.echo(f"Categories filter: {categories}")
        click.echo(f"Reranking: {'disabled' if no_rerank else 'enabled'}")
    
    try:
        # Parse categories
        category_filter = None
        if categories:
            category_filter = [cat.strip() for cat in categories.split(',')]
        
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Execute search
        results = indexer.search_mental_models(
            query=query,
            persona_id=persona_id,
            k=top_k,
            use_reranking=not no_rerank,
            min_confidence_score=min_confidence,
            filter_by_categories=category_filter,
            return_scores=show_scores
        )
        
        if not results:
            click.echo("❌ No mental models found matching the query.")
            return
        
        # Display results
        if output_format == 'json':
            # JSON output
            json_results = []
            for i, result in enumerate(results):
                if show_scores and isinstance(result, tuple):
                    model_result, score = result
                    json_results.append({
                        'rank': i + 1,
                        'score': score,
                        'name': model_result.name,
                        'description': model_result.description,
                        'steps': model_result.steps,
                        'categories': model_result.categories,
                        'confidence_score': model_result.confidence_score,
                        'frequency': model_result.frequency
                    })
                else:
                    json_results.append({
                        'rank': i + 1,
                        'name': result.name,
                        'description': result.description,
                        'steps': result.steps,
                        'categories': result.categories,
                        'confidence_score': result.confidence_score,
                        'frequency': result.frequency
                    })
            
            click.echo(json.dumps(json_results, indent=2))
        
        else:
            # Text output
            click.echo(f"✅ Found {len(results)} mental models:")
            click.echo("")
            
            for i, result in enumerate(results):
                if show_scores and isinstance(result, tuple):
                    model_result, score = result
                    click.echo(f"🧠 {i+1}. {model_result.name} (Score: {score:.3f})")
                else:
                    click.echo(f"🧠 {i+1}. {result.name}")
                
                # Get the actual result object
                actual_result = result[0] if show_scores and isinstance(result, tuple) else result
                
                click.echo(f"   Description: {actual_result.description}")
                if actual_result.categories:
                    click.echo(f"   Categories: {actual_result.get_categories_string()}")
                click.echo(f"   Confidence: {actual_result.confidence_score:.2f}")
                
                if verbose:
                    click.echo(f"   Steps ({len(actual_result.steps)}):")
                    for step in actual_result.steps:
                        click.echo(f"     • {step}")
                else:
                    click.echo(f"   Steps: {len(actual_result.steps)} steps")
                
                click.echo("")  # Blank line between results
    
    except Exception as e:
        click.echo(f"❌ Search error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--query', required=True, help='Search query')
@click.option('--top-k', default=5, help='Number of results to return (default: 5)')
@click.option('--min-confidence', default=0.0, help='Minimum confidence score (default: 0.0)')
@click.option('--category', help='Filter by specific category')
@click.option('--conviction-level', type=click.Choice(['very_high', 'high', 'moderate', 'moderate_low', 'low']), 
              help='Filter by conviction level')
@click.option('--no-evidence', is_flag=True, help='Exclude supporting evidence')
@click.option('--no-rerank', is_flag=True, help='Disable reranking')
@click.option('--show-scores', is_flag=True, help='Show relevance scores')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def search_core_beliefs(ctx, persona_id: str, query: str, top_k: int, min_confidence: float,
                       category: Optional[str], conviction_level: Optional[str], no_evidence: bool,
                       no_rerank: bool, show_scores: bool, output_format: str, verbose: bool):
    """Search core beliefs knowledge base."""
    
    if verbose:
        click.echo(f"Searching core beliefs for persona: {persona_id}")
        click.echo(f"Query: {query}")
        click.echo(f"Top-K: {top_k}")
        click.echo(f"Min confidence: {min_confidence}")
        if category:
            click.echo(f"Category filter: {category}")
        if conviction_level:
            click.echo(f"Conviction level: {conviction_level}")
        click.echo(f"Include evidence: {'no' if no_evidence else 'yes'}")
        click.echo(f"Reranking: {'disabled' if no_rerank else 'enabled'}")
    
    try:
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Execute search
        results = indexer.search_core_beliefs(
            query=query,
            persona_id=persona_id,
            k=top_k,
            use_reranking=not no_rerank,
            min_confidence_score=min_confidence,
            filter_by_category=category,
            conviction_level=conviction_level,
            include_evidence=not no_evidence,
            return_scores=show_scores
        )
        
        if not results:
            click.echo("❌ No core beliefs found matching the query.")
            return
        
        # Display results
        if output_format == 'json':
            # JSON output
            json_results = []
            for i, result in enumerate(results):
                if show_scores and isinstance(result, tuple):
                    belief_result, score = result
                    json_results.append({
                        'rank': i + 1,
                        'score': score,
                        'statement': belief_result.statement,
                        'category': belief_result.category,
                        'supporting_evidence': belief_result.supporting_evidence,
                        'confidence_score': belief_result.confidence_score,
                        'frequency': belief_result.frequency
                    })
                else:
                    json_results.append({
                        'rank': i + 1,
                        'statement': result.statement,
                        'category': result.category,
                        'supporting_evidence': result.supporting_evidence,
                        'confidence_score': result.confidence_score,
                        'frequency': result.frequency
                    })
            
            click.echo(json.dumps(json_results, indent=2))
        
        else:
            # Text output
            click.echo(f"✅ Found {len(results)} core beliefs:")
            click.echo("")
            
            for i, result in enumerate(results):
                if show_scores and isinstance(result, tuple):
                    belief_result, score = result
                    click.echo(f"💭 {i+1}. (Score: {score:.3f})")
                else:
                    click.echo(f"💭 {i+1}.")
                
                # Get the actual result object
                actual_result = result[0] if show_scores and isinstance(result, tuple) else result
                
                click.echo(f"   Statement: {actual_result.statement}")
                click.echo(f"   Category: {actual_result.category}")
                click.echo(f"   Confidence: {actual_result.confidence_score:.2f} ({actual_result.get_confidence_level()})")
                
                if not no_evidence and actual_result.supporting_evidence:
                    if verbose:
                        click.echo(f"   Supporting Evidence:")
                        for evidence in actual_result.supporting_evidence:
                            click.echo(f"     • {evidence}")
                    else:
                        click.echo(f"   Evidence: {len(actual_result.supporting_evidence)} items")
                
                click.echo("")  # Blank line between results
    
    except Exception as e:
        click.echo(f"❌ Search error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.pass_context
def list_indexes(ctx, persona_id: str, output_format: str):
    """List all knowledge indexes for a persona."""
    
    try:
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Get statistics
        stats = indexer.get_multi_knowledge_statistics(persona_id)
        
        if output_format == 'json':
            click.echo(json.dumps(stats, indent=2))
        else:
            click.echo(f"📊 Knowledge Indexes for Persona: {persona_id}")
            click.echo("")
            
            if not stats.get('multi_knowledge_enabled'):
                click.echo("❌ Multi-knowledge system not enabled or failed to initialize")
                if 'error' in stats:
                    click.echo(f"   Error: {stats['error']}")
                return
            
            # Vector store stats
            vector_stats = stats.get('vector_store_stats', {})
            collections = vector_stats.get('collections', {})
            
            for knowledge_type, collection_info in collections.items():
                if collection_info.get('collection_exists'):
                    click.echo(f"✅ {knowledge_type.replace('_', ' ').title()}")
                    click.echo(f"   Documents: {collection_info.get('documents_indexed', 'unknown')}")
                    if 'directory_size_mb' in collection_info:
                        click.echo(f"   Size: {collection_info['directory_size_mb']} MB")
                    if 'last_indexed' in collection_info:
                        click.echo(f"   Last indexed: {collection_info['last_indexed']}")
                else:
                    click.echo(f"❌ {knowledge_type.replace('_', ' ').title()} (not built)")
                click.echo("")
            
            # Summary
            summary = vector_stats.get('summary', {})
            if summary:
                click.echo(f"📈 Summary:")
                click.echo(f"   Total collections: {summary.get('total_collections', 0)}")
                click.echo(f"   Total documents: {summary.get('total_documents', 0)}")
                click.echo(f"   Total size: {summary.get('total_size_mb', 0):.1f} MB")
    
    except Exception as e:
        click.echo(f"❌ Error listing indexes: {e}", err=True)


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--knowledge-type', type=click.Choice(['mental_models', 'core_beliefs', 'all']), 
              default='all', help='Knowledge type to clear (default: all)')
@click.option('--cache-only', is_flag=True, help='Clear only cache, not indexes')
@click.confirmation_option(prompt='Are you sure you want to clear the knowledge base(s)?')
@click.pass_context
def clear(ctx, persona_id: str, knowledge_type: str, cache_only: bool):
    """Clear knowledge base indexes and/or cache."""
    
    try:
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        if cache_only:
            # Clear cache only
            if hasattr(indexer, 'multi_knowledge_cache') and indexer.multi_knowledge_cache:
                if knowledge_type == 'all':
                    indexer.multi_knowledge_cache.clear_cache()
                    click.echo("✅ Cleared all knowledge type caches")
                else:
                    kt = KnowledgeType.from_string(knowledge_type)
                    indexer.multi_knowledge_cache.clear_cache(kt)
                    click.echo(f"✅ Cleared cache for {kt.display_name.lower()}")
            else:
                click.echo("❌ Multi-knowledge cache not initialized")
        else:
            # Clear indexes (this would require additional methods in MultiKnowledgeStore)
            click.echo("⚠️ Index clearing not yet implemented - use rebuild flag when building indexes")
    
    except Exception as e:
        click.echo(f"❌ Error clearing knowledge base: {e}", err=True)


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--rebuild', is_flag=True, help='Rebuild existing index from scratch')
@click.option('--validate/--no-validate', default=True, help='Validate JSON schema (default: validate)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def build_mental_models_auto(ctx, persona_id: str, rebuild: bool, validate: bool, verbose: bool):
    """Build mental models knowledge base from latest Phase 1 artifact (auto-discovery)."""
    
    if verbose:
        click.echo(f"Auto-building mental models index for persona: {persona_id}")
        click.echo(f"Rebuild: {rebuild}")
        click.echo(f"Validate: {validate}")
        click.echo("🔍 Auto-discovering latest artifact...")
    
    try:
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Build mental models index with auto-discovery
        with click.progressbar(length=100, label="Building mental models index") as bar:
            bar.update(20)  # Artifact discovery
            
            result = indexer.build_mental_models_index_auto(
                persona_id=persona_id,
                rebuild=rebuild,
                validate=validate
            )
            
            bar.update(100)  # Complete
        
        # Display results
        if result.success:
            click.echo(f"✅ Mental models index built successfully!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings and verbose:
                click.echo("Warnings:")
                for warning in result.warnings:
                    click.echo(f"   - {warning}")
        elif result.partial_success:
            click.echo(f"⚠️ Mental models index built with warnings!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings:
                click.echo("Warnings:")
                for warning in result.warnings[:10]:  # Show first 10 warnings
                    click.echo(f"   - {warning}")
                if len(result.warnings) > 10:
                    click.echo(f"   ... and {len(result.warnings) - 10} more warnings")
        else:
            click.echo(f"❌ Failed to build mental models index")
            if result.errors:
                click.echo("Errors:")
                for error in result.errors:
                    click.echo(f"   - {error}")
    
    except Exception as e:
        click.echo(f"❌ Critical error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


@knowledge.command()
@click.option('--persona-id', required=True, help='Persona identifier')
@click.option('--rebuild', is_flag=True, help='Rebuild existing index from scratch')
@click.option('--validate/--no-validate', default=True, help='Validate JSON schema (default: validate)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def build_core_beliefs_auto(ctx, persona_id: str, rebuild: bool, validate: bool, verbose: bool):
    """Build core beliefs knowledge base from latest Phase 1 artifact (auto-discovery)."""
    
    if verbose:
        click.echo(f"Auto-building core beliefs index for persona: {persona_id}")
        click.echo(f"Rebuild: {rebuild}")
        click.echo(f"Validate: {validate}")
        click.echo("🔍 Auto-discovering latest artifact...")
    
    try:
        # Initialize indexer
        indexer = KnowledgeIndexer(
            settings=ctx.obj['settings'],
            persona_manager=ctx.obj['persona_manager'],
            persona_id=persona_id
        )
        
        # Build core beliefs index with auto-discovery
        with click.progressbar(length=100, label="Building core beliefs index") as bar:
            bar.update(20)  # Artifact discovery
            
            result = indexer.build_core_beliefs_index_auto(
                persona_id=persona_id,
                rebuild=rebuild,
                validate=validate
            )
            
            bar.update(100)  # Complete
        
        # Display results
        if result.success:
            click.echo(f"✅ Core beliefs index built successfully!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings and verbose:
                click.echo("Warnings:")
                for warning in result.warnings:
                    click.echo(f"   - {warning}")
        elif result.partial_success:
            click.echo(f"⚠️ Core beliefs index built with warnings!")
            click.echo(f"   Documents indexed: {result.documents_indexed}")
            if result.documents_processed > result.documents_indexed:
                skipped = result.documents_processed - result.documents_indexed
                click.echo(f"   Documents skipped: {skipped} (validation errors)")
            click.echo(f"   Processing time: {result.indexing_duration_seconds:.2f}s")
            if result.index_size_mb > 0:
                click.echo(f"   Index size: {result.index_size_mb:.1f} MB")
            if result.warnings:
                click.echo("Warnings:")
                for warning in result.warnings[:10]:  # Show first 10 warnings
                    click.echo(f"   - {warning}")
                if len(result.warnings) > 10:
                    click.echo(f"   ... and {len(result.warnings) - 10} more warnings")
        else:
            click.echo(f"❌ Failed to build core beliefs index")
            if result.errors:
                click.echo("Errors:")
                for error in result.errors:
                    click.echo(f"   - {error}")
    
    except Exception as e:
        click.echo(f"❌ Critical error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    knowledge()