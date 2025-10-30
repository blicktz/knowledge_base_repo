"""
MCP Server for Influencer Persona Agent - SIMPLIFIED VERSION.
Provides ONLY 3 data retrieval tools. Query analysis and style
are handled in Skills, not here!
"""

import json
import asyncio
from typing import Any, Dict, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from dk_rag.config.settings import Settings
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.core.persona_manager import PersonaManager


class PersonaMCPServer:
    """Simplified MCP Server - 3 tools for data retrieval only."""

    def __init__(self):
        # Load settings
        config_path = "dk_rag/config/persona_config.yaml"
        self.settings = Settings(config_path=config_path)

        # Initialize MCP server
        self.server = Server("persona-agent")

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
        @self.server.call_tool()
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
        @self.server.call_tool()
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
        @self.server.call_tool()
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

        # Register tool metadata
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="retrieve_mental_models",
                    description="Retrieve step-by-step frameworks and mental models for process-oriented queries. Returns structured frameworks with name, description, and steps.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Process-oriented search query (10-20 words with context, e.g., 'customer acquisition strategies for AI SAAS startup')"
                            },
                            "persona_id": {
                                "type": "string",
                                "description": "Persona identifier (e.g., 'dan_kennedy', 'greg_startup')"
                            }
                        },
                        "required": ["query", "persona_id"]
                    }
                ),
                Tool(
                    name="retrieve_core_beliefs",
                    description="Retrieve philosophical principles and core beliefs for opinion-based queries. Returns belief statements with category and supporting evidence.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Principle-oriented search query (8-15 words, e.g., 'beliefs about customer retention and loyalty')"
                            },
                            "persona_id": {
                                "type": "string",
                                "description": "Persona identifier (e.g., 'dan_kennedy', 'greg_startup')"
                            }
                        },
                        "required": ["query", "persona_id"]
                    }
                ),
                Tool(
                    name="retrieve_transcripts",
                    description="Retrieve real examples, stories, and anecdotes from transcripts. Use for factual queries and concrete evidence.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Example-specific search query (10-20 words with specific context, e.g., 'successful lead magnet examples with conversion rates')"
                            },
                            "persona_id": {
                                "type": "string",
                                "description": "Persona identifier (e.g., 'dan_kennedy', 'greg_startup')"
                            }
                        },
                        "required": ["query", "persona_id"]
                    }
                )
            ]

    async def run(self):
        """Run the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Entry point for MCP server."""
    server = PersonaMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
