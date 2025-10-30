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
    """Configure logging for MCP stdio transport - file only, no console output."""
    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up file-based logging only
    log_dir = Path("./logs/mcp_server")
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "mcp_server.log",
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


class PersonaMCPServer:
    """Simplified MCP Server - 3 tools for data retrieval only."""

    def __init__(self):
        # Configure logging for MCP stdio transport (file-only, no console)
        configure_mcp_logging()

        # Load settings using default config (path-independent)
        self.settings = Settings.from_default_config()

        # Initialize MCP server with ERROR log level to prevent logging leakage
        self.mcp = FastMCP("persona-agent", log_level="ERROR")

        # Initialize persona manager
        self.persona_manager = PersonaManager(self.settings)

        # Cache for persona-specific knowledge indexers
        self._knowledge_indexers: Dict[str, KnowledgeIndexer] = {}

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
            try:
                indexer = self.get_knowledge_indexer(persona_id)
                results = indexer.search_mental_models(
                    query=query,
                    persona_id=persona_id,
                    k=3,
                    use_reranking=True
                )

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
            try:
                indexer = self.get_knowledge_indexer(persona_id)
                results = indexer.search_core_beliefs(
                    query=query,
                    persona_id=persona_id,
                    k=3,
                    use_reranking=True
                )

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
            try:
                indexer = self.get_knowledge_indexer(persona_id)

                # Initialize advanced retrieval pipeline if not already done
                if not indexer.advanced_pipeline:
                    indexer._init_retrieval_components()

                # Use advanced pipeline for transcript retrieval
                results = await asyncio.to_thread(
                    indexer.advanced_pipeline.retrieve,
                    query=query,
                    k=3,
                    return_scores=True
                )

                formatted_results = []
                for doc, score in results:
                    formatted_results.append({
                        "content": doc.page_content,
                        "document_id": doc.metadata.get("document_id", ""),
                        "chunk_id": doc.metadata.get("chunk_id", ""),
                        "score": float(score) if score else 0.0,
                        "metadata": doc.metadata
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
        self.mcp.run(transport="stdio")


def main():
    """Entry point for MCP server."""
    server = PersonaMCPServer()
    server.run()


if __name__ == "__main__":
    main()
