"""
MCP Server for Influencer Persona Agent.

This module provides a Model Context Protocol (MCP) server that exposes
3 data retrieval tools for accessing influencer persona knowledge bases:
- retrieve_mental_models: Step-by-step frameworks and processes
- retrieve_core_beliefs: Philosophical principles and beliefs
- retrieve_transcripts: Real examples, stories, and anecdotes

The server uses a simplified architecture where it only handles data retrieval,
while query analysis and linguistic style are embedded in Claude Code Skills.
"""

from dk_rag.mcp_server.persona_mcp_server import PersonaMCPServer, main

__all__ = ["PersonaMCPServer", "main"]
