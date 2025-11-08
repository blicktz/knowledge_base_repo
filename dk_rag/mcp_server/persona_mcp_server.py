"""
MCP Server for Influencer Persona Agent - SIMPLIFIED VERSION.
Provides ONLY 3 data retrieval tools. Query analysis and style
are handled in Skills, not here!
"""

import json
import asyncio
import logging
import logging.handlers
from pathlib import Path
from typing import Dict
from mcp.server import FastMCP
from mcp.types import TextContent

from dk_rag.config.settings import Settings
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.core.persona_manager import PersonaManager


def configure_mcp_logging():
    """Configure logging for MCP stdio transport - file only, no console output.

    Returns:
        Path: Absolute path to the log file
    """
    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up file-based logging only - use absolute path
    # Get project root (3 levels up from this file: mcp_server -> dk_rag -> project_root)
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs" / "mcp_server"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "mcp_server.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    return log_file_path


class PersonaMCPServer:
    """Simplified MCP Server - 3 tools for data retrieval only."""

    def __init__(self):
        # Configure logging for MCP stdio transport (file-only, no console)
        self.log_file_path = configure_mcp_logging()

        # Load settings using default config (path-independent)
        self.settings = Settings.from_default_config()

        # Initialize MCP server with INFO log level for observability
        self.mcp = FastMCP("persona-agent", log_level="INFO")

        # Initialize persona manager
        self.persona_manager = PersonaManager(self.settings)

        # Cache for persona-specific knowledge indexers
        self._knowledge_indexers: Dict[str, KnowledgeIndexer] = {}

        # Set up logger for tool call tracking
        self.logger = logging.getLogger(__name__)

        # Register tools
        self._register_tools()

    def get_knowledge_indexer(self, persona_id: str) -> KnowledgeIndexer:
        """Get or create knowledge indexer for a persona."""
        if persona_id not in self._knowledge_indexers:
            self._knowledge_indexers[persona_id] = KnowledgeIndexer(
                settings=self.settings,
                persona_manager=self.persona_manager,
                persona_id=persona_id
            )
        return self._knowledge_indexers[persona_id]

    def _register_tools(self):
        """Register the 3 data retrieval tools."""

        # Tool 1: Retrieve Mental Models
        @self.mcp.tool()
        async def retrieve_mental_models(query: str, persona_id: str) -> list[TextContent]:
            """
            Retrieve step-by-step frameworks and mental models.

            Use for "how-to" questions that need process guidance.
            Query should be 10-20 words with rich context.
            """
            self.logger.info(f"[TOOL CALL] retrieve_mental_models | persona_id={persona_id} | query={query}")
            try:
                indexer = self.get_knowledge_indexer(persona_id)
                results = indexer.search_mental_models(
                    query=query,
                    persona_id=persona_id,
                    k=3,
                    use_reranking=True
                )

                self.logger.info(f"[TOOL SUCCESS] retrieve_mental_models | results_count={len(results)}")
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "name": result.name if hasattr(result, 'name') else "",
                        "description": result.description if hasattr(result, 'description') else "",
                        "steps": result.steps if hasattr(result, 'steps') else [],
                        "categories": result.categories if hasattr(result, 'categories') else [],
                        "confidence_score": result.confidence_score if hasattr(result, 'confidence_score') else 0.0
                    })

                output = {
                    "tool": "retrieve_mental_models",
                    "persona_id": persona_id,
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, indent=2)
                )]

            except Exception as e:
                self.logger.error(f"[TOOL ERROR] retrieve_mental_models | error={str(e)}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": "retrieve_mental_models",
                        "persona_id": persona_id,
                        "query": query
                    }, indent=2)
                )]

        # Tool 2: Retrieve Core Beliefs
        @self.mcp.tool()
        async def retrieve_core_beliefs(query: str, persona_id: str) -> list[TextContent]:
            """
            Retrieve philosophical principles and core beliefs.

            Use for "why" questions and opinion-based queries.
            Query should be 8-15 words focused on principles.
            """
            self.logger.info(f"[TOOL CALL] retrieve_core_beliefs | persona_id={persona_id} | query={query}")
            try:
                indexer = self.get_knowledge_indexer(persona_id)
                results = indexer.search_core_beliefs(
                    query=query,
                    persona_id=persona_id,
                    k=3,
                    use_reranking=True
                )

                self.logger.info(f"[TOOL SUCCESS] retrieve_core_beliefs | results_count={len(results)}")
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "statement": result.statement if hasattr(result, 'statement') else "",
                        "category": result.category if hasattr(result, 'category') else "",
                        "confidence_score": result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                        "frequency": result.frequency if hasattr(result, 'frequency') else 0,
                        "supporting_evidence": result.supporting_evidence if hasattr(result, 'supporting_evidence') else []
                    })

                output = {
                    "tool": "retrieve_core_beliefs",
                    "persona_id": persona_id,
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, indent=2)
                )]

            except Exception as e:
                self.logger.error(f"[TOOL ERROR] retrieve_core_beliefs | error={str(e)}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": "retrieve_core_beliefs",
                        "persona_id": persona_id,
                        "query": query
                    }, indent=2)
                )]

        # Tool 3: Retrieve Transcripts
        @self.mcp.tool()
        async def retrieve_transcripts(query: str, persona_id: str) -> list[TextContent]:
            """
            Retrieve real examples, stories, and anecdotes from transcripts.

            Use for factual queries and concrete evidence.
            Query should be 10-20 words with specific context.
            """
            self.logger.info(f"[TOOL CALL] retrieve_transcripts | persona_id={persona_id} | query={query}")
            try:
                indexer = self.get_knowledge_indexer(persona_id)

                # Get the advanced retrieval pipeline (initializes if needed)
                pipeline = indexer.get_advanced_retrieval_pipeline(persona_id)

                if not pipeline:
                    # Fallback error if Phase 2 not available
                    self.logger.error(f"[TOOL ERROR] retrieve_transcripts | error=Pipeline not available")
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Advanced retrieval pipeline not available. Phase 2 may not be enabled.",
                            "tool": "retrieve_transcripts",
                            "persona_id": persona_id,
                            "query": query
                        }, indent=2)
                    )]

                # Use advanced pipeline for transcript retrieval
                results = await asyncio.to_thread(
                    pipeline.retrieve,
                    query=query,
                    k=3,
                    use_hyde=True,
                    use_hybrid=True,
                    use_reranking=True,
                    return_scores=True
                )

                self.logger.info(f"[TOOL SUCCESS] retrieve_transcripts | results_count={len(results)}")
                formatted_results = []
                for result in results:
                    # Handle tuple format (doc, score) when return_scores=True
                    if isinstance(result, tuple) and len(result) == 2:
                        doc, score = result
                        formatted_results.append({
                            "content": doc.page_content,
                            "document_id": doc.metadata.get("document_id", ""),
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            "score": float(score) if score else 0.0,
                            "metadata": doc.metadata
                        })
                    elif hasattr(result, 'page_content'):
                        # Document without score
                        formatted_results.append({
                            "content": result.page_content,
                            "document_id": result.metadata.get("document_id", ""),
                            "chunk_id": result.metadata.get("chunk_id", ""),
                            "score": 0.0,
                            "metadata": result.metadata
                        })

                output = {
                    "tool": "retrieve_transcripts",
                    "persona_id": persona_id,
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, indent=2)
                )]

            except Exception as e:
                self.logger.error(f"[TOOL ERROR] retrieve_transcripts | error={str(e)}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": "retrieve_transcripts",
                        "persona_id": persona_id,
                        "query": query
                    }, indent=2)
                )]

    def run(self):
        """Run the MCP server using stdio transport."""
        self.logger.info("=" * 60)
        self.logger.info("MCP Server Starting - persona-agent")
        self.logger.info(f"Log file: {self.log_file_path.absolute()}")
        self.logger.info("Available tools: retrieve_mental_models, retrieve_core_beliefs, retrieve_transcripts")
        self.logger.info("=" * 60)
        self.mcp.run(transport="stdio")


def main():
    """Entry point for MCP server."""
    server = PersonaMCPServer()
    server.run()


if __name__ == "__main__":
    main()
