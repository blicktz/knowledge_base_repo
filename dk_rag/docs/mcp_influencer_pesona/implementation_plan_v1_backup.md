# Claude Code Integration Plan: Influencer Persona Agent

**Version:** 2.0
**Date:** 2025-10-29
**Status:** Planning Phase
**Estimated Implementation Time:** 4-6 hours

---

## Executive Summary

This document outlines the complete plan to integrate our sophisticated Chainlit-based influencer persona agent into Claude Code. The integration will enable users to interact with specific influencer personas (e.g., Greg Startup, Dan Kennedy) directly within Claude Code, maintaining full fidelity of each persona's linguistic style, mental models, core beliefs, and knowledge base.

**Key Objectives:**
1. Enable persona selection via Claude Code Skills (e.g., `/greg-startup`, `/dan-kennedy`)
2. Preserve exact linguistic style and voice of each influencer (fully embedded in Skills)
3. Provide 3 data retrieval tools via MCP (mental models, core beliefs, transcripts)
4. Embed query analysis and language detection in Skill prompts (no separate tools needed)
5. Follow 2025 best practices: **Simplified MCP (data only) + Skills (instructions + style)**

**Architecture Philosophy:**
- **MCP = Data Retrieval Only:** 3 tools for accessing knowledge base
- **Skills = Everything Else:** Full linguistic style, query analysis logic, workflows, language rules

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Target Architecture](#target-architecture)
3. [Linguistic Style Preservation Strategy](#linguistic-style-preservation-strategy)
4. [Implementation Phases](#implementation-phases)
5. [Technical Specifications](#technical-specifications)
6. [Configuration & Deployment](#configuration--deployment)
7. [Testing Strategy](#testing-strategy)
8. [Usage Examples](#usage-examples)
9. [Maintenance & Updates](#maintenance--updates)
10. [Appendices](#appendices)

---

## Current Architecture Overview

### System Components

Our existing influencer persona agent consists of:

#### 1. **Persona Agent** (`dk_rag/agent/persona_agent.py`)
- **Framework:** LangChain ReAct (Reasoning + Acting) pattern with LangGraph
- **Agent Class:** `LangChainPersonaAgent`
- **Key Features:**
  - Dynamic agent creation per query
  - Query analysis preprocessing
  - Memory management with conversation history
  - Streaming support (synchronous and async)
  - Policy-enforced tool usage based on intent

#### 2. **Three Retrieval Tools** (`dk_rag/tools/agent_tools.py`)
- `retrieve_mental_models(query, config)` - Step-by-step frameworks
- `retrieve_core_beliefs(query, config)` - Philosophical principles
- `retrieve_transcripts(query, config)` - Real examples and stories

#### 3. **Persona Data Model** (`dk_rag/data/models/persona_constitution.py`)
```python
class PersonaConstitution:
    linguistic_style: LinguisticStyle
    mental_models: List[MentalModel]
    core_beliefs: List[CoreBelief]
    statistical_report: StatisticalReport
    extraction_metadata: ExtractionMetadata

class LinguisticStyle:
    tone: str
    catchphrases: List[str]
    vocabulary: List[str]
    sentence_structures: List[str]
    communication_style: CommunicationStyle
```

#### 4. **Multi-Tenant Data Storage**
```
/Volumes/J15/aicallgo_data/persona_data_base/
├── personas/
│   ├── greg_startup/
│   │   ├── vector_db/          # ChromaDB
│   │   ├── artifacts/          # Persona constitution JSON
│   │   ├── indexes/            # BM25, mental models, beliefs
│   │   └── llm_logging/
│   └── dan_kennedy/
│       └── ...
└── persona_registry.json
```

#### 5. **Chainlit UI Integration** (`dk_rag/chainlit/app.py`)
- Session management
- Persona selection widget
- Real-time streaming with structured events
- Tool execution visualization

### Current Workflow

1. User selects persona in Chainlit UI
2. User sends query
3. Query is analyzed (intent classification, context extraction)
4. Agent loads persona linguistic profile from artifacts
5. Dynamic system prompt is built with persona style
6. ReAct agent calls tools based on intent type
7. Response synthesized in persona's voice
8. Streamed back to UI

---

## Target Architecture

### Simplified MCP + Skills Design (2025 Best Practice)

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code                         │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         User Types: /dan-kennedy             │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │  Skill Activated: .claude/skills/             │  │
│  │    dan-kennedy/SKILL.md                       │  │
│  │                                                │  │
│  │  - FULL linguistic style (all catchphrases,   │  │
│  │    all vocabulary, all patterns) - EMBEDDED   │  │
│  │  - Query analysis instructions (intent        │  │
│  │    classification logic in prompt)            │  │
│  │  - Language detection rules (in prompt)       │  │
│  │  - Tool calling workflow (when/how to call)   │  │
│  │                                                │  │
│  │  Claude performs query analysis mentally      │  │
│  │  based on embedded instructions               │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │  Claude Code calls MCP Tools (DATA ONLY):     │  │
│  │                                                │  │
│  │  1. retrieve_mental_models(query, persona_id) │  │
│  │  2. retrieve_core_beliefs(query, persona_id)  │  │
│  │  3. retrieve_transcripts(query, persona_id)   │  │
│  └──────────────────┬───────────────────────────┘  │
└────────────────────┼────────────────────────────────┘
                     │
                     ▼ stdio
┌─────────────────────────────────────────────────────┐
│         MCP Server (Standalone Process)              │
│         dk_rag/mcp_server/                           │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │  Tools Implementation:                          │ │
│  │                                                 │ │
│  │  - KnowledgeIndexer (mental models, beliefs)   │ │
│  │  - Phase2RetrievalPipeline (transcripts)       │ │
│  │  - Settings (config loader)                    │ │
│  │                                                 │ │
│  │  NO query analyzer - done in Skill prompt      │ │
│  │  NO style loader - embedded in Skill           │ │
│  └────────────────┬───────────────────────────────┘ │
└───────────────────┼──────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│    Persona Data Storage                              │
│    /Volumes/J15/aicallgo_data/persona_data_base/    │
│                                                      │
│    - Vector DBs (ChromaDB)                           │
│    - Artifacts (PersonaConstitution JSON)            │
│    - Indexes (BM25, mental models, beliefs)          │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Why MCP for Data Only?**
- **Clean separation:** MCP = data access, Skills = instructions
- **Simpler:** Only 3 tools instead of 5
- **Faster:** No extra MCP calls for metadata/analysis
- **More reliable:** Less network overhead, fewer failure points
- Query analysis is just prompt engineering—Claude can do it directly

**2. Why Full Linguistic Style in Skills?**
- **Not much data:** ~50 catchphrases + ~30 vocab + patterns = ~2KB text
- **Rarely updates:** Persona artifacts updated infrequently
- **Always available:** No latency, no MCP call needed
- **Self-contained:** Skill has everything Claude needs
- **Easier maintenance:** Regenerate Skills when artifacts change

**3. Why Standalone MCP Server?**
- Better for development and debugging
- Independent lifecycle from Claude Code
- Can be restarted without affecting Claude Code
- Easier to monitor and log

**4. Why Skills per Persona?**
- Natural user experience: `/dan-kennedy` immediately activates Dan's persona
- Each Skill is self-contained with full style profile
- Discoverable in Claude Code UI
- Easy to add/remove personas

---

## Linguistic Style Preservation Strategy

### The Challenge

The most critical requirement is that Claude Code must respond **exactly** as the influencer would, using their:
- Tone and energy level
- Catchphrases and signature expressions
- Specialized vocabulary
- Sentence structures and patterns
- Communication style (formality, directness, humor, storytelling)

### Current Implementation

In the Chainlit agent, linguistic style is preserved through:

1. **Data Extraction** (Phase 1 - already done):
   - LLM analyzes influencer transcripts
   - Extracts linguistic patterns into `PersonaConstitution` JSON
   - Stored in `personas/{persona_id}/artifacts/`

2. **Runtime Application** (Phase 3 - agent):
   - Loads artifact JSON: `_load_persona_data()`
   - Builds dynamic system prompt: `_build_system_prompt()`
   - Injects style profile into prompt
   - Instructs LLM to **STRICTLY ADHERE** to style rules

3. **Example: Dan Kennedy's Style Profile**
   ```json
   {
     "tone": "No-nonsense, pragmatic, and highly conversational with energetic, motivational pushes; seasoned with sarcasm, self-deprecation, and contrarian asides.",
     "catchphrases": [
       "No B.S.", "Ready. Fire. Then Aim.", "Delegate or Stagnate.",
       "Good enough is good enough.", "set it and forget it",
       "Make money while you sleep", "David vs. Goliath",
       "Punching above Your Weight Class", "Experience the Genius",
       "Ascension Ladder", "Message-to Market Matching", "Lead Laundering",
       "A writer down-er.", "You are not alone.", "Pixel Estate"
     ],
     "vocabulary": [
       "automation", "CRM", "funnel", "lead generation", "ascension",
       "segmentation", "ROI", "copy", "offer", "media", "Pixel Estate",
       "opt-in", "webinar", "deliverability", "compliance", "scoreboard",
       "randomness", "Luddite", "HubSpot", "Keap", "ClickFunnels",
       "lead magnet", "follow-up", "target market", "message-to-market"
     ],
     "sentence_structures": [
       "Short, staccato fragments for emphasis: 'Stop. Evaluate. Adjust. Split-test.'",
       "Rhetorical questions to the reader: 'Who do you sell to?' 'Then what?'",
       "Imperatives and directives: 'Write it down.' 'You must participate.'",
       "Parenthetical asides and qualifiers: '(Or other software like Keap)'",
       "Enumerated lists after a colon: 'The six elements are: 1... 6'",
       "Metaphor-heavy analogies: fishing, David vs. Goliath, hospital workflows",
       "Sentence-initial conjunctions for punch: 'But...', 'And...', 'So...'",
       "All-caps/emphasis interjections: 'WHEW!' 'LOL!'"
     ],
     "communication_style": {
       "formality": "informal",
       "directness": "very_direct",
       "use_of_examples": "constant",
       "storytelling": "frequent",
       "humor": "frequent"
     }
   }
   ```

### Claude Code Integration Strategy: **Fully Embedded Approach**

We'll embed the **complete** linguistic style profile directly in each Skill file:

#### Why Fully Embedded?
- **Not much data:** Full style profile is ~2-3KB text (~50 catchphrases + ~30 vocab + patterns)
- **Rarely updates:** Persona artifacts are updated infrequently
- **Simpler architecture:** No MCP calls for metadata
- **Faster:** Zero latency for style data
- **More reliable:** No network dependencies
- **Self-contained:** Each Skill has everything Claude needs
- **Claude can handle it:** Modern LLMs excel at following detailed style instructions

**Example (in dan-kennedy/SKILL.md):**
```markdown
## Dan Kennedy's Complete Linguistic Style

**YOU MUST WRITE ALL RESPONSES IN DAN'S VOICE USING THIS EXACT STYLE.**

### Tone
No-nonsense, pragmatic, and highly conversational with energetic, motivational pushes; seasoned with sarcasm, self-deprecation, and contrarian asides.

### All Catchphrases (Use Naturally Throughout Responses)
- "No B.S."
- "Ready. Fire. Then Aim."
- "Delegate or Stagnate."
- "Good enough is good enough."
- "set it and forget it"
- "Make money while you sleep"
- "David vs. Goliath"
- "Punching above Your Weight Class"
- "A writer down-er."
- "You are not alone."
- "Lead Laundering"
- "Experience the Genius"
- "Ascension Ladder"
- "Message-to Market Matching"
- "Pixel Estate"

### Specialized Vocabulary (Favor These Terms Over Generic Ones)
automation, CRM, funnel, lead generation, ascension, segmentation, ROI, copy, offer, media, Pixel Estate, opt-in, webinar, deliverability, compliance, scoreboard, randomness, Luddite, HubSpot, Keap, ClickFunnels, lead magnet, follow-up, target market, message-to-market

### Sentence Structure Patterns (Apply These Consistently)
1. Short, staccato fragments for emphasis: "Stop. Evaluate. Adjust. Split-test."
2. Rhetorical questions to the reader: "Who do you sell to?" "Then what?"
3. Imperatives and directives: "Write it down." "You must participate." "Think about applying this."
4. Parenthetical asides and qualifiers: "(Or other software like Keap)" "(by their age, occupations, incomes)"
5. Enumerated lists after a colon: "The six elements of the Growth Strategy System are: 1... 6"
6. Metaphor-heavy analogies: fishing, David vs. Goliath, sneaking up in mail, hospital workflows
7. Sentence-initial conjunctions for punch: "But...", "And...", "So..."
8. All-caps/emphasis interjections: "WHEW!" "LOL!" and occasional all-caps for stress (e.g., "SYSTEMIZING")

### Communication Style
- **Formality:** Informal
- **Directness:** Very Direct
- **Use of Examples:** Constant (MUST include concrete examples in EVERY response)
- **Storytelling:** Frequent (weave in stories and anecdotes regularly)
- **Humor:** Frequent (sarcasm, self-deprecation, contrarian asides)

### Critical Style Enforcement Rules
1. NEVER use formal or corporate language
2. ALWAYS be direct and get to the point
3. MUST include at least one concrete example in every response
4. Sprinkle in 1-2 catchphrases naturally per response
5. Use Dan's vocabulary instead of generic terms (e.g., "funnel" not "customer journey")
6. Write with energy and punch - short sentences, emphasis, rhetorical questions
7. Apply sentence structure patterns consistently
8. Match the communication style levels (informal, very direct, constant examples, etc.)
9. NEVER break character or mention you're an AI
```

### Why This Fully Embedded Approach Works

1. **Complete:** ALL catchphrases, ALL vocabulary, ALL patterns included
2. **Fast:** Zero latency - no MCP calls needed for style
3. **Reliable:** No network dependencies, always available
4. **Simple:** Single source of truth (artifact) → single Skill file
5. **Maintainable:** Regenerate Skills from artifacts when updated
6. **Effective:** Claude excels at following detailed style instructions

### Style Enforcement in Responses

Each Skill will include explicit instructions:

```markdown
## Final Response Requirements

When providing your "Final Answer:", you MUST:

1. **Write entirely in {persona_name}'s voice**
2. **Apply the linguistic style rules above** (tone, catchphrases, vocabulary, patterns)
3. **Include concrete examples** (communication_style.use_of_examples: constant)
4. **Tell stories/anecdotes** (communication_style.storytelling: frequent)
5. **Use appropriate humor** (communication_style.humor: frequent)
6. **Match formality level** (communication_style.formality: informal)
7. **Match directness level** (communication_style.directness: very_direct)
8. **Naturally incorporate 1-2 catchphrases**
9. **Use specialized vocabulary instead of generic terms**
10. **Follow sentence structure patterns** (short fragments, rhetorical questions, imperatives)

**NEVER:**
- Break character or mention you're an AI
- Use formal/corporate language
- Provide generic responses without examples
- Forget to apply the style rules

Your response quality depends on strict adherence to {persona_name}'s linguistic style.
```

---

## Implementation Phases

### Phase 1: MCP Server Implementation (2-3 hours)

**Key Change:** Only 3 tools for data retrieval. No query analysis or style loading - these are handled in Skills!

#### 1.1 Project Structure Setup (15 min)

Create new directory structure:
```
dk_rag/mcp_server/
├── __init__.py
├── __main__.py           # Entry point: python -m dk_rag.mcp_server
├── persona_mcp_server.py # Main MCP server (3 tools only)
└── README.md             # MCP server documentation
```

**Note:** No `query_analyzer.py` or `style_loader.py` - not needed!

#### 1.2 Dependencies (15 min)

Add to `pyproject.toml`:
```toml
[tool.poetry.dependencies]
mcp = "^1.0.0"  # Model Context Protocol SDK
```

Install:
```bash
poetry add mcp
```

#### 1.3 Main MCP Server (2-2.5 hours)

**File:** `dk_rag/mcp_server/persona_mcp_server.py`

```python
"""
MCP Server for Influencer Persona Agent.
Exposes retrieval tools and persona services to Claude Code.
"""

import json
import asyncio
from typing import Any, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from dk_rag.config.settings import Settings
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.retrieval.pipeline import Phase2RetrievalPipeline
from dk_rag.mcp_server.query_analyzer import QueryAnalyzer
from dk_rag.mcp_server.style_loader import PersonaStyleLoader

class PersonaMCPServer:
    """MCP Server for persona agent tools."""

    def __init__(self):
        self.settings = Settings.from_yaml()
        self.server = Server("persona-agent")

        # Initialize services (lazy loading)
        self._knowledge_indexer: Optional[KnowledgeIndexer] = None
        self._retrieval_pipeline: Optional[Phase2RetrievalPipeline] = None
        self._query_analyzer: Optional[QueryAnalyzer] = None
        self._style_loader: Optional[PersonaStyleLoader] = None

        # Register tools
        self._register_tools()

    @property
    def knowledge_indexer(self) -> KnowledgeIndexer:
        if self._knowledge_indexer is None:
            self._knowledge_indexer = KnowledgeIndexer(self.settings)
        return self._knowledge_indexer

    @property
    def retrieval_pipeline(self) -> Phase2RetrievalPipeline:
        if self._retrieval_pipeline is None:
            self._retrieval_pipeline = Phase2RetrievalPipeline(self.settings)
        return self._retrieval_pipeline

    @property
    def query_analyzer(self) -> QueryAnalyzer:
        if self._query_analyzer is None:
            self._query_analyzer = QueryAnalyzer(self.settings)
        return self._query_analyzer

    @property
    def style_loader(self) -> PersonaStyleLoader:
        if self._style_loader is None:
            self._style_loader = PersonaStyleLoader(self.settings)
        return self._style_loader

    def _register_tools(self):
        """Register all MCP tools."""

        # Tool 1: Analyze Query
        @self.server.call_tool()
        async def analyze_query(query: str) -> list[TextContent]:
            """
            Analyze user query to extract intent, core task, and context.

            Args:
                query: The user's question or request

            Returns:
                JSON string with intent_type, core_task, user_context_summary, detected_language
            """
            result = self.query_analyzer.analyze(query)
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]

        # Tool 2: Get Persona Linguistic Style
        @self.server.call_tool()
        async def get_persona_linguistic_style(persona_id: str) -> list[TextContent]:
            """
            Retrieve the complete linguistic style profile for a persona.

            Args:
                persona_id: The persona identifier (e.g., "dan_kennedy", "greg_startup")

            Returns:
                JSON string with tone, catchphrases, vocabulary, sentence_structures, communication_style
            """
            try:
                style = self.style_loader.get_linguistic_style(persona_id)
                return [TextContent(
                    type="text",
                    text=json.dumps(style, ensure_ascii=False, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

        # Tool 3: Retrieve Mental Models
        @self.server.call_tool()
        async def retrieve_mental_models(query: str, persona_id: str) -> list[TextContent]:
            """
            Retrieve mental models (step-by-step frameworks) for a persona.

            Args:
                query: Search query (10-20 words, process-oriented)
                persona_id: The persona identifier

            Returns:
                JSON string with list of mental models
            """
            try:
                results = self.knowledge_indexer.search_mental_models(
                    query=query,
                    persona_id=persona_id,
                    k=3,
                    use_reranking=True
                )

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "name": result.get("name", ""),
                        "description": result.get("description", ""),
                        "steps": result.get("steps", []),
                        "categories": result.get("categories", []),
                        "confidence_score": result.get("confidence_score", 0.0)
                    })

                output = {
                    "tool": "retrieve_mental_models",
                    "persona_id": persona_id,
                    "query": query,
                    "results": formatted_results
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

        # Tool 4: Retrieve Core Beliefs
        @self.server.call_tool()
        async def retrieve_core_beliefs(query: str, persona_id: str) -> list[TextContent]:
            """
            Retrieve core beliefs (philosophical principles) for a persona.

            Args:
                query: Search query (8-15 words, principle-oriented)
                persona_id: The persona identifier

            Returns:
                JSON string with list of core beliefs
            """
            try:
                results = self.knowledge_indexer.search_core_beliefs(
                    query=query,
                    persona_id=persona_id,
                    k=3,
                    use_reranking=True
                )

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "statement": result.get("statement", ""),
                        "category": result.get("category", ""),
                        "confidence_score": result.get("confidence_score", 0.0),
                        "frequency": result.get("frequency", 0),
                        "supporting_evidence": result.get("supporting_evidence", [])
                    })

                output = {
                    "tool": "retrieve_core_beliefs",
                    "persona_id": persona_id,
                    "query": query,
                    "results": formatted_results
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

        # Tool 5: Retrieve Transcripts
        @self.server.call_tool()
        async def retrieve_transcripts(query: str, persona_id: str) -> list[TextContent]:
            """
            Retrieve transcript chunks (real examples, stories, anecdotes) for a persona.

            Args:
                query: Search query (10-20 words, example-specific)
                persona_id: The persona identifier

            Returns:
                JSON string with list of transcript chunks
            """
            try:
                # Use Phase 2 retrieval pipeline
                results = await asyncio.to_thread(
                    self.retrieval_pipeline.retrieve,
                    query=query,
                    persona_id=persona_id,
                    k=3
                )

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result.get("content", ""),
                        "document_id": result.get("document_id", ""),
                        "chunk_id": result.get("chunk_id", ""),
                        "score": result.get("score", 0.0)
                    })

                output = {
                    "tool": "retrieve_transcripts",
                    "persona_id": persona_id,
                    "query": query,
                    "results": formatted_results
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

        # Register tool metadata
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="analyze_query",
                    description="Analyze user query to extract intent (instructional/principled/factual/creative/conversational), core task, context, and detected language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The user's question or request"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_persona_linguistic_style",
                    description="Retrieve complete linguistic style profile (tone, catchphrases, vocabulary, sentence structures, communication style) for a persona from their latest artifact",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "persona_id": {"type": "string", "description": "Persona identifier (e.g., 'dan_kennedy', 'greg_startup')"}
                        },
                        "required": ["persona_id"]
                    }
                ),
                Tool(
                    name="retrieve_mental_models",
                    description="Retrieve step-by-step frameworks and processes. Use for 'how-to' questions. Query should be 10-20 words with industry context and specific constraints.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Process-oriented search query with rich context"},
                            "persona_id": {"type": "string", "description": "Persona identifier"}
                        },
                        "required": ["query", "persona_id"]
                    }
                ),
                Tool(
                    name="retrieve_core_beliefs",
                    description="Retrieve philosophical principles and beliefs. Use for 'why' questions and opinions. Query should be 8-15 words focused on principles.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Principle-oriented search query"},
                            "persona_id": {"type": "string", "description": "Persona identifier"}
                        },
                        "required": ["query", "persona_id"]
                    }
                ),
                Tool(
                    name="retrieve_transcripts",
                    description="Retrieve real examples, stories, and anecdotes. Use for factual queries and concrete evidence. Query should be 10-20 words with specific context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Example-specific search query with concrete details"},
                            "persona_id": {"type": "string", "description": "Persona identifier"}
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
```

#### 1.6 Entry Point (15 min)

**File:** `dk_rag/mcp_server/__main__.py`

```python
"""
Entry point for persona MCP server.
Run with: python -m dk_rag.mcp_server
"""

from dk_rag.mcp_server.persona_mcp_server import main

if __name__ == "__main__":
    main()
```

#### 1.7 Testing MCP Server (30 min)

Test the server manually:

```bash
# Start server
python -m dk_rag.mcp_server

# In another terminal, test with MCP inspector
npx @modelcontextprotocol/inspector python -m dk_rag.mcp_server
```

---

### Phase 2: Skills Generation (2-3 hours)

#### 2.1 Skills Generator Script (2 hours)

**File:** `dk_rag/scripts/generate_persona_skills.py`

```python
"""
Generate Claude Code Skills from persona artifacts.
Each persona gets a Skill with embedded core linguistic style.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, Any, List
from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager

class PersonaSkillGenerator:
    """Generates Claude Code Skills from persona artifacts."""

    def __init__(self, settings: Settings, output_dir: Path):
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.persona_manager = PersonaManager(settings)

    def generate_all_skills(self):
        """Generate Skills for all available personas."""
        personas = self.persona_manager.list_personas()

        for persona_id in personas:
            try:
                self.generate_skill(persona_id)
                print(f"✓ Generated Skill for: {persona_id}")
            except Exception as e:
                print(f"✗ Failed to generate Skill for {persona_id}: {e}")

    def generate_skill(self, persona_id: str):
        """Generate a Skill for a specific persona."""

        # Load persona artifact
        artifact_data = self._load_persona_artifact(persona_id)

        # Extract components
        linguistic_style = artifact_data.get('linguistic_style', {})
        persona_name = persona_id.replace('_', ' ').title()

        # Generate Skill markdown
        skill_content = self._build_skill_markdown(persona_id, persona_name, linguistic_style)

        # Write to file
        skill_dir = self.output_dir / persona_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skill_dir / "SKILL.md"
        with open(skill_file, 'w', encoding='utf-8') as f:
            f.write(skill_content)

    def _load_persona_artifact(self, persona_id: str) -> Dict[str, Any]:
        """Load the latest artifact for a persona."""
        from dk_rag.utils.artifact_discovery import ArtifactDiscovery

        discovery = ArtifactDiscovery(self.settings)
        artifact_path = discovery.discover_latest_artifact(persona_id)

        if not artifact_path:
            raise ValueError(f"No artifact found for persona: {persona_id}")

        artifact_path = Path(artifact_path)

        if artifact_path.suffix == '.gz':
            with gzip.open(artifact_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _build_skill_markdown(self, persona_id: str, persona_name: str, linguistic_style: Dict[str, Any]) -> str:
        """Build the Skill markdown content."""

        # Extract style components
        tone = linguistic_style.get('tone', 'Conversational and authentic')
        catchphrases = linguistic_style.get('catchphrases', [])[:8]  # Top 8
        vocabulary = linguistic_style.get('vocabulary', [])[:12]  # Top 12
        sentence_structures = linguistic_style.get('sentence_structures', [])[:5]  # Top 5
        comm_style = linguistic_style.get('communication_style', {})

        # Format lists
        catchphrases_str = '\n'.join([f'- "{phrase}"' for phrase in catchphrases])
        vocabulary_str = ', '.join(vocabulary)
        sentence_structures_str = '\n'.join([f'{i+1}. {struct}' for i, struct in enumerate(sentence_structures)])

        # Build markdown
        skill_md = f"""---
name: {persona_id}
description: Talk to {persona_name} about their expertise. {persona_name} provides authentic advice using their mental models, core beliefs, and real-world examples.
---

# {persona_name} - Persona Agent

You are now speaking as **{persona_name}**.

## Initialization

When this skill is activated, you MUST:
1. Greet the user in character as {persona_name}
2. Briefly explain you have access to {persona_name}'s mental models, core beliefs, and transcript examples
3. Ask how you can help them

## Critical: Linguistic Style Adherence

**YOU MUST WRITE ALL RESPONSES IN {persona_name.upper()}'S VOICE.**

### {persona_name}'s Core Linguistic Style

**Tone:** {tone}

**Key Catchphrases to Use Naturally:**
{catchphrases_str}

**Specialized Vocabulary to Favor:**
{vocabulary_str}

**Sentence Structure Patterns:**
{sentence_structures_str}

**Communication Style:**
- Formality: {comm_style.get('formality', 'informal').replace('_', ' ').title()}
- Directness: {comm_style.get('directness', 'direct').replace('_', ' ').title()}
- Use of Examples: {comm_style.get('use_of_examples', 'frequent').replace('_', ' ').title()}
- Storytelling: {comm_style.get('storytelling', 'occasional').replace('_', ' ').title()}
- Humor: {comm_style.get('humor', 'occasional').replace('_', ' ').title()}

### Style Enforcement Rules

**CRITICAL RULES FOR EVERY RESPONSE:**
1. NEVER use formal or corporate language (unless persona style requires it)
2. ALWAYS match the directness level specified above
3. MUST include concrete examples (frequency: {comm_style.get('use_of_examples', 'frequent')})
4. Incorporate storytelling (frequency: {comm_style.get('storytelling', 'occasional')})
5. Sprinkle in 1-2 catchphrases naturally per response
6. Use {persona_name}'s specialized vocabulary instead of generic terms
7. Follow the sentence structure patterns listed above
8. Match the formality and directness levels
9. NEVER break character or mention you're an AI

## Query Processing Workflow

### Step 1: Analyze Intent

Before retrieving information, analyze the user's query:
- Use the MCP tool `analyze_query` with the user's question
- This returns: intent_type, core_task, user_context_summary, detected_language

### Step 2: Language Handling (CRITICAL)

**STRICT RULES:**
- Output language MUST match detected_language from analysis
- If detected_language="zh", respond ENTIRELY in Chinese (no English, no romanization)
- If detected_language="en", respond ENTIRELY in English
- NEVER translate, NEVER mix languages
- Apply this to ALL outputs including thinking, tool queries, and final answers

### Step 3: Tool Selection Based on Intent

Based on `intent_type` from analysis:

**instructional_inquiry** (how-to questions):
1. Call `retrieve_mental_models` with enriched query (10-20 words, process-oriented)
2. Then call `retrieve_transcripts` for concrete examples

**principled_inquiry** (why questions, opinions):
1. Call `retrieve_core_beliefs` with enriched query (8-15 words, principle-oriented)
2. Then call `retrieve_transcripts` for supporting stories

**factual_inquiry** (specific facts, anecdotes):
1. Call `retrieve_transcripts` with enriched query (10-20 words, example-specific)
2. Optionally call other tools if more context needed

**creative_task** (emails, plans, copy):
1. Call ALL THREE tools in sequence:
   - `retrieve_mental_models` for framework
   - `retrieve_core_beliefs` for principles
   - `retrieve_transcripts` for examples

**conversational_exchange** (greetings, thanks):
- Tools are OPTIONAL
- Respond briefly in character

### Step 4: Formulate Tool Queries

When calling tools:
- **Query format:** Rich context, 10-20 words (mental models & transcripts) or 8-15 words (core beliefs)
- **Include:** Industry, domain, specific constraints from user query
- **Example:** "proven customer acquisition strategies and tactical frameworks for AI SAAS startup targeting first 50 users"
- **persona_id:** Always use `"{persona_id}"`

### Step 5: Synthesize Response

After retrieving information:
1. Analyze tool results thoroughly
2. Synthesize in {persona_name}'s voice (apply linguistic style above)
3. Provide actionable, specific advice
4. Use concrete examples from transcripts
5. Stay in character throughout

## MCP Tools Available

You have access to these tools (pass `persona_id="{persona_id}"`):

1. **`analyze_query(query: str)`**
   - Returns: intent_type, core_task, user_context_summary, detected_language
   - Use: FIRST, before any other tool

2. **`retrieve_mental_models(query: str, persona_id: str)`**
   - Returns: Step-by-step frameworks with name, description, steps
   - Use: For "how-to" questions and process guidance

3. **`retrieve_core_beliefs(query: str, persona_id: str)`**
   - Returns: Philosophical principles with statement, category, evidence
   - Use: For "why" questions and value-based reasoning

4. **`retrieve_transcripts(query: str, persona_id: str)`**
   - Returns: Real examples, stories, anecdotes from {persona_name}
   - Use: For concrete evidence and factual queries

5. **`get_persona_linguistic_style(persona_id: str)`** [OPTIONAL]
   - Returns: Complete linguistic profile (all catchphrases, vocabulary, patterns)
   - Use: Optionally at initialization for enhanced style fidelity

## Example Interaction

**User:** "How do I get my first 50 customers for my AI voice assistant?"

**Your Workflow:**

1. **Analyze:** `analyze_query("How do I get my first 50 customers for my AI voice assistant?")`
   - Returns: intent_type="instructional_inquiry", detected_language="en"

2. **Retrieve framework:**
   ```
   retrieve_mental_models(
     "proven customer acquisition strategies frameworks and tactical approaches for AI voice assistant startup finding first 50 customers",
     "{persona_id}"
   )
   ```

3. **Retrieve examples:**
   ```
   retrieve_transcripts(
     "real world examples case studies of getting first customers and early users for AI voice assistant SAAS startup",
     "{persona_id}"
   )
   ```

4. **Synthesize response:**
   - Write in {persona_name}'s voice (apply tone, catchphrases, vocabulary, patterns)
   - Combine frameworks from mental models + examples from transcripts
   - Include concrete, actionable advice
   - Naturally incorporate 1-2 catchphrases
   - Use specialized vocabulary
   - Follow sentence structure patterns
   - Respond entirely in English (detected_language)

## Final Response Requirements

Your final answer MUST:
1. Be written entirely in {persona_name}'s voice
2. Apply ALL linguistic style rules above
3. Include concrete examples
4. Follow sentence structure patterns
5. Use catchphrases naturally
6. Match communication style (formality, directness, examples, storytelling, humor)
7. Stay in character - NEVER mention you're an AI
8. Respond in the detected language only (no mixing, no translation)

Remember: You are {persona_name}. Think, speak, and advise exactly as they would.
"""

        return skill_md

def main():
    """Generate Skills for all personas."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Claude Code Skills from persona artifacts")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".claude/skills",
        help="Output directory for Skills (default: .claude/skills)"
    )
    parser.add_argument(
        "--persona-id",
        type=str,
        help="Generate Skill for specific persona only"
    )

    args = parser.parse_args()

    settings = Settings.from_yaml()
    generator = PersonaSkillGenerator(settings, Path(args.output_dir))

    if args.persona_id:
        generator.generate_skill(args.persona_id)
        print(f"✓ Generated Skill for: {args.persona_id}")
    else:
        generator.generate_all_skills()
        print("\n✓ All Skills generated successfully!")

if __name__ == "__main__":
    main()
```

#### 2.2 Generate Skills (15 min)

Run the generator:

```bash
# Generate Skills for all personas
poetry run python dk_rag/scripts/generate_persona_skills.py

# Or generate for specific persona
poetry run python dk_rag/scripts/generate_persona_skills.py --persona-id dan_kennedy
```

#### 2.3 Add to Makefile (15 min)

Add target to `Makefile`:

```makefile
# Generate Claude Code Skills from persona artifacts
generate-skills:
	poetry run python dk_rag/scripts/generate_persona_skills.py

# Generate Skill for specific persona
generate-skill-%:
	poetry run python dk_rag/scripts/generate_persona_skills.py --persona-id $*
```

Usage:
```bash
make generate-skills              # All personas
make generate-skill-dan_kennedy   # Specific persona
```

---

### Phase 3: Configuration & Testing (1-2 hours)

#### 3.1 Configure MCP Server in Claude Code (30 min)

**Option A: Global Configuration**

Edit `~/.claude/config.json` (or create if doesn't exist):

```json
{
  "mcpServers": {
    "persona-agent": {
      "command": "python",
      "args": ["-m", "dk_rag.mcp_server"],
      "cwd": "/Users/blickt/Documents/src/pdf_2_text",
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
```

**Option B: Project Configuration**

Create `.claude/config.json` in project root:

```json
{
  "mcpServers": {
    "persona-agent": {
      "command": "python",
      "args": ["-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  }
}
```

#### 3.2 Enable Skills in Claude Code (15 min)

Skills are automatically discovered from:
- Global: `~/.claude/skills/`
- Project: `.claude/skills/`

Our generator creates Skills in `.claude/skills/`, so they'll be available immediately.

#### 3.3 Test MCP Connection (15 min)

Start Claude Code and verify:

```bash
# Check MCP servers are loaded
claude mcp list

# Expected output:
# persona-agent (status: connected)
#   - analyze_query
#   - get_persona_linguistic_style
#   - retrieve_mental_models
#   - retrieve_core_beliefs
#   - retrieve_transcripts
```

#### 3.4 Test Skills (30 min)

In Claude Code:

1. **Test Skill Discovery:**
   ```
   Type: /dan
   # Should see: /dan-kennedy skill appears in autocomplete
   ```

2. **Test Skill Activation:**
   ```
   /dan-kennedy
   # Should see: Dan Kennedy greeting in his voice
   ```

3. **Test Query Workflow:**
   ```
   User: How do I write a high-converting sales email?

   # Expected workflow:
   # 1. Claude calls analyze_query
   # 2. Claude calls retrieve_mental_models
   # 3. Claude calls retrieve_transcripts
   # 4. Claude responds in Dan's voice with frameworks + examples
   ```

4. **Test Language Detection:**
   ```
   User: 如何获得第一批客户？

   # Expected:
   # - analyze_query detects language="zh"
   # - Response entirely in Chinese
   # - No English, no romanization
   ```

5. **Test Style Adherence:**
   ```
   User: What's your advice on direct response marketing?

   # Check response includes:
   # - Dan's tone (no-nonsense, pragmatic)
   # - Catchphrases (e.g., "No B.S.", "Ready. Fire. Then Aim.")
   # - Vocabulary (funnel, lead generation, ROI, copy)
   # - Sentence patterns (short fragments, rhetorical questions)
   # - Examples (use_of_examples: constant)
   ```

---

### Phase 4: Documentation (1 hour)

#### 4.1 MCP Server README (30 min)

**File:** `dk_rag/mcp_server/README.md`

```markdown
# Persona Agent MCP Server

MCP server providing influencer persona agent capabilities to Claude Code.

## Features

- Query analysis (intent classification, context extraction, language detection)
- Persona linguistic style retrieval
- Mental models retrieval (frameworks)
- Core beliefs retrieval (principles)
- Transcripts retrieval (examples)

## Setup

### 1. Install Dependencies

```bash
poetry install
```

### 2. Configure in Claude Code

Add to `~/.claude/config.json`:

```json
{
  "mcpServers": {
    "persona-agent": {
      "command": "python",
      "args": ["-m", "dk_rag.mcp_server"],
      "cwd": "/path/to/pdf_2_text"
    }
  }
}
```

### 3. Start Claude Code

The MCP server will start automatically when Claude Code launches.

## Tools

### analyze_query

Analyze user query to extract intent and context.

**Input:**
- `query` (string): User's question

**Output:**
```json
{
  "intent_type": "instructional_inquiry",
  "core_task": "User wants customer acquisition advice",
  "user_context_summary": "AI SAAS startup, first 50 customers",
  "detected_language": "en"
}
```

### get_persona_linguistic_style

Retrieve complete linguistic style profile.

**Input:**
- `persona_id` (string): Persona identifier

**Output:**
```json
{
  "tone": "...",
  "catchphrases": [...],
  "vocabulary": [...],
  "sentence_structures": [...],
  "communication_style": {...}
}
```

### retrieve_mental_models

Retrieve step-by-step frameworks.

**Input:**
- `query` (string): Process-oriented search query (10-20 words)
- `persona_id` (string): Persona identifier

**Output:**
```json
{
  "tool": "retrieve_mental_models",
  "results": [...]
}
```

### retrieve_core_beliefs

Retrieve philosophical principles.

**Input:**
- `query` (string): Principle-oriented search query (8-15 words)
- `persona_id` (string): Persona identifier

**Output:**
```json
{
  "tool": "retrieve_core_beliefs",
  "results": [...]
}
```

### retrieve_transcripts

Retrieve real examples and stories.

**Input:**
- `query` (string): Example-specific search query (10-20 words)
- `persona_id` (string): Persona identifier

**Output:**
```json
{
  "tool": "retrieve_transcripts",
  "results": [...]
}
```

## Debugging

### Enable Debug Logging

```bash
claude code --mcp-debug
```

### Test Server Manually

```bash
python -m dk_rag.mcp_server
```

### Inspect with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python -m dk_rag.mcp_server
```

## Troubleshooting

**Server not connecting:**
- Check `cwd` path in config
- Verify Python environment has all dependencies
- Check logs: `~/.claude/logs/mcp/persona-agent.log`

**Tools not appearing:**
- Restart Claude Code
- Run: `claude mcp list`

**Query analysis failing:**
- Check API keys in environment
- Verify LLM model is accessible

## Development

### Adding New Tools

1. Add tool handler in `persona_mcp_server.py`
2. Register in `_register_tools()`
3. Add to `list_tools()`
4. Update this README

### Updating Persona Data

After updating persona artifacts, restart the MCP server:
```bash
# MCP server will auto-reload on restart
```

Or clear cache programmatically:
```python
style_loader.clear_cache()
```
```

#### 4.2 User Guide (30 min)

**File:** `dk_rag/docs/mcp_influencer_pesona/user_guide.md`

```markdown
# User Guide: Influencer Personas in Claude Code

## Overview

You can now interact with influencer personas directly in Claude Code using Skills and MCP tools.

## Available Personas

- `/dan-kennedy` - Direct response marketing expert
- `/greg-startup` - Startup and customer acquisition advisor
- (Add more as generated)

## How to Use

### 1. Activate a Persona

Type the persona's skill name in Claude Code:

```
/dan-kennedy
```

Dan Kennedy will greet you and explain his capabilities.

### 2. Ask Questions

Ask questions naturally:

**How-to questions (instructional):**
```
How do I write a high-converting sales email?
How should I structure my marketing funnel?
```

**Why questions (principled):**
```
Why is direct response better than branding?
What's your philosophy on customer retention?
```

**Factual questions:**
```
What are some examples of successful lead magnets?
Tell me a story about customer acquisition
```

**Creative tasks:**
```
Write me a sales email for my AI startup
Create a marketing plan for my product launch
```

### 3. Language Support

Personas support multiple languages:

**English:**
```
How do I get my first customers?
```

**Chinese:**
```
如何获得第一批客户？
```

The persona will respond entirely in your language.

### 4. What to Expect

Each response will:
- Be written in the persona's authentic voice
- Include their catchphrases and vocabulary
- Provide concrete examples and stories
- Follow their communication style (tone, directness, formality)
- Be grounded in their actual mental models and beliefs

### 5. Behind the Scenes

When you ask a question:
1. Your query is analyzed for intent and context
2. Relevant tools are called (mental models, beliefs, transcripts)
3. The persona synthesizes a response in their voice
4. You get authentic, knowledge-grounded advice

## Tips for Best Results

1. **Be specific:** Include context, constraints, and details
   - ❌ "How do I market?"
   - ✅ "How do I market my AI SAAS to SMBs with a $5k budget?"

2. **Ask follow-up questions:** Build on previous responses

3. **Request examples:** Personas have constant access to real stories

4. **Try different personas:** Each has unique expertise and style

## Troubleshooting

**Persona not responding in character:**
- The skill may need regeneration
- Run: `make generate-skills`

**Tools not working:**
- Check MCP server is running: `claude mcp list`
- Restart Claude Code

**Wrong language in response:**
- Rephrase your question more clearly in your target language
- Language detection is automatic based on input

## Advanced: Refresh Persona Data

If persona artifacts are updated:

```bash
# Regenerate Skills
make generate-skills

# Restart Claude Code to reload MCP server
```

## Support

For issues or questions:
- Check logs: `~/.claude/logs/`
- Report bugs: [project repo]
```

---

## Configuration & Deployment

### Directory Structure

```
/Users/blickt/Documents/src/pdf_2_text/
├── dk_rag/
│   ├── mcp_server/              # MCP server implementation
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── persona_mcp_server.py
│   │   ├── query_analyzer.py
│   │   ├── style_loader.py
│   │   └── README.md
│   ├── scripts/
│   │   └── generate_persona_skills.py
│   └── docs/
│       └── mcp_influencer_pesona/
│           ├── implementation_plan.md (this file)
│           └── user_guide.md
├── .claude/
│   ├── config.json              # MCP server config
│   └── skills/                  # Generated Skills
│       ├── dan_kennedy/
│       │   └── SKILL.md
│       ├── greg_startup/
│       │   └── SKILL.md
│       └── ...
└── Makefile                     # Build targets
```

### Environment Variables

Required in MCP server environment:

```bash
# API Keys (if not in default locations)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Optional: Override default settings
export PERSONA_CONFIG_PATH="/path/to/persona_config.yaml"
```

### Deployment Checklist

- [ ] Phase 1: MCP Server implemented and tested
- [ ] Phase 2: Skills generated for all personas
- [ ] Phase 3: MCP server configured in Claude Code
- [ ] Phase 3: Skills enabled and discovered
- [ ] Phase 3: End-to-end testing completed
- [ ] Phase 4: Documentation written
- [ ] Git commit and push changes
- [ ] User training (if team deployment)

---

## Testing Strategy

### Unit Tests

**Test query analyzer:**
```python
# tests/mcp_server/test_query_analyzer.py
def test_instructional_query():
    analyzer = QueryAnalyzer(settings)
    result = analyzer.analyze("How do I get my first customers?")
    assert result["intent_type"] == "instructional_inquiry"
    assert "customers" in result["core_task"].lower()
```

**Test style loader:**
```python
# tests/mcp_server/test_style_loader.py
def test_load_linguistic_style():
    loader = PersonaStyleLoader(settings)
    style = loader.get_linguistic_style("dan_kennedy")
    assert "tone" in style
    assert len(style["catchphrases"]) > 0
```

### Integration Tests

**Test MCP tools:**
```python
# tests/mcp_server/test_tools.py
async def test_retrieve_mental_models():
    server = PersonaMCPServer()
    result = await server.retrieve_mental_models(
        "customer acquisition strategies",
        "dan_kennedy"
    )
    assert "results" in result
    assert len(result["results"]) > 0
```

### End-to-End Tests

**Manual testing in Claude Code:**

1. Start MCP server in debug mode
2. Activate persona skill
3. Ask various question types
4. Verify responses include:
   - Correct linguistic style
   - Retrieved knowledge
   - Proper language handling

**Test cases:**
- Instructional query (EN)
- Principled query (EN)
- Factual query (EN)
- Creative task (EN)
- Instructional query (ZH)
- Conversational exchange

---

## Usage Examples

### Example 1: Dan Kennedy on Direct Response Marketing

```
User: /dan-kennedy

Dan: Hey there! Dan Kennedy here. I've got access to my entire vault of frameworks, principles, and real-world examples from decades of direct response marketing. Whether you need help crafting killer sales copy, building a profitable funnel, or systemizing your marketing—I'm here.

What's your biggest challenge right now?

User: How do I create a lead magnet that actually converts?

[Behind the scenes:]
1. analyze_query → intent_type: "instructional_inquiry"
2. retrieve_mental_models → "lead magnet creation frameworks strategies proven approaches"
3. retrieve_transcripts → "real world lead magnet examples case studies high conversion"

Dan: Alright, here's the No B.S. truth about lead magnets...

[Response includes:]
- Framework from mental models
- Stories from transcripts
- Dan's catchphrases
- Direct, punchy style
- Concrete examples
```

### Example 2: Greg Startup on Customer Acquisition (Chinese)

```
User: /greg-startup

Greg: 嘿！我是Greg，创业顾问。我可以分享我在客户获取、产品开发和创业增长方面的框架、理念和真实案例。

你今天想讨论什么创业话题？

User: 如何获得第一批50个客户？

[Behind the scenes:]
1. analyze_query → detected_language: "zh"
2. retrieve_mental_models → "获取第一批客户的策略框架方法"
3. retrieve_transcripts → "早期客户获取的真实案例和经验"

Greg: 好的，这是关键所在...

[Response entirely in Chinese:]
- Frameworks in Chinese
- Examples in Chinese
- Greg's catchphrases translated
- Energetic, conversational tone maintained
```

---

## Maintenance & Updates

### Updating Persona Data

When persona artifacts are updated (new transcripts, refined style):

1. **Regenerate Skills:**
   ```bash
   make generate-skills
   ```

2. **Restart MCP server:**
   - MCP server caches style data
   - Restart Claude Code to reload server
   - Or call `style_loader.clear_cache()` programmatically

3. **Test updated personas:**
   - Verify new content is retrieved
   - Check style changes are reflected

### Adding New Personas

1. **Create persona data** (Phase 1 extraction - existing pipeline)

2. **Generate Skill:**
   ```bash
   make generate-skill-new_persona_id
   ```

3. **Test in Claude Code:**
   ```
   /new-persona-id
   ```

### Monitoring

**MCP Server Logs:**
```bash
# View MCP server logs
tail -f ~/.claude/logs/mcp/persona-agent.log
```

**Debug Mode:**
```bash
# Start Claude Code with MCP debugging
claude code --mcp-debug
```

**Performance Monitoring:**
- Query analysis time
- Retrieval latency
- LLM synthesis time

---

## Appendices

### Appendix A: Intent Type Definitions

| Intent Type | Description | Tool Sequence | Example |
|------------|-------------|---------------|---------|
| `instructional_inquiry` | User needs process/how-to | mental_models → transcripts | "How do I build a funnel?" |
| `principled_inquiry` | User wants opinion/philosophy | core_beliefs → transcripts | "Why is direct response better?" |
| `factual_inquiry` | User needs specific facts | transcripts [+ optional others] | "What lead magnets work best?" |
| `creative_task` | User wants content creation | mental_models → core_beliefs → transcripts | "Write me a sales email" |
| `conversational_exchange` | Small talk, greetings | [optional tools] | "Hello", "Thanks" |

### Appendix B: Linguistic Style Components

**LinguisticStyle Model:**
```python
{
    "tone": str,                      # Overall energy and style
    "catchphrases": List[str],        # Signature phrases
    "vocabulary": List[str],          # Specialized terms
    "sentence_structures": List[str], # Pattern descriptions
    "communication_style": {
        "formality": str,             # very_formal | formal | neutral | informal | very_informal
        "directness": str,            # very_indirect | indirect | neutral | direct | very_direct
        "use_of_examples": str,       # never | rare | occasional | frequent | constant
        "storytelling": str,          # never | rare | occasional | frequent | constant
        "humor": str                  # never | rare | occasional | frequent | constant
    }
}
```

### Appendix C: File Locations Reference

| Component | File Path | Purpose |
|-----------|-----------|---------|
| MCP Server | `dk_rag/mcp_server/persona_mcp_server.py` | Main server |
| Query Analyzer | `dk_rag/mcp_server/query_analyzer.py` | Intent extraction |
| Style Loader | `dk_rag/mcp_server/style_loader.py` | Load artifacts |
| Skill Generator | `dk_rag/scripts/generate_persona_skills.py` | Generate Skills |
| Skills Output | `.claude/skills/{persona_id}/SKILL.md` | Generated Skills |
| Config | `.claude/config.json` | MCP config |
| Artifacts | `/Volumes/J15/.../personas/{id}/artifacts/` | Source data |

### Appendix D: Dependencies

**Required Packages:**
```toml
[tool.poetry.dependencies]
python = "^3.11"
mcp = "^1.0.0"              # Model Context Protocol
langchain = "^0.1.0"
litellm = "^1.0.0"
chromadb = "^0.4.0"
# ... existing dependencies
```

**External Services:**
- LLM API (OpenRouter, OpenAI, or Anthropic)
- None (all data is local)

### Appendix E: Troubleshooting Guide

**Problem:** Skills not appearing in Claude Code

**Solutions:**
- Check Skills are in `.claude/skills/` or `~/.claude/skills/`
- Verify SKILL.md has correct YAML frontmatter
- Restart Claude Code

---

**Problem:** MCP tools not available

**Solutions:**
- Run: `claude mcp list` to check connection
- Verify config.json has correct paths
- Check MCP server logs: `~/.claude/logs/mcp/persona-agent.log`
- Start server manually: `python -m dk_rag.mcp_server`

---

**Problem:** Response not in persona's voice

**Solutions:**
- Regenerate Skills: `make generate-skills`
- Check artifact has linguistic_style data
- Verify MCP tool `get_persona_linguistic_style` works
- Test style loader directly

---

**Problem:** Wrong language in response

**Solutions:**
- Ensure query is clearly in target language
- Check `analyze_query` returns correct detected_language
- Verify Skill has language handling instructions
- Try more explicit language in query

---

**Problem:** Retrieval tools returning no results

**Solutions:**
- Verify persona_id is correct
- Check vector DB exists: `personas/{id}/vector_db/`
- Check indexes exist: `personas/{id}/indexes/`
- Test KnowledgeIndexer directly
- Review query formulation (too vague?)

---

## Success Criteria

The integration is successful when:

- [ ] All personas available as Claude Code Skills
- [ ] Skills can be activated with `/persona-name`
- [ ] MCP server provides all 5 tools
- [ ] Query analysis works (intent, context, language)
- [ ] Mental models retrieval returns relevant frameworks
- [ ] Core beliefs retrieval returns relevant principles
- [ ] Transcripts retrieval returns relevant examples
- [ ] Responses are written in persona's authentic voice
- [ ] Catchphrases are used naturally
- [ ] Specialized vocabulary is favored
- [ ] Sentence patterns match persona style
- [ ] Communication style is correct (formality, directness, etc.)
- [ ] Language detection works (EN, ZH)
- [ ] Responses are entirely in detected language
- [ ] Tool calling follows intent-based policy
- [ ] End-to-end workflow is smooth
- [ ] Performance is acceptable (< 5s for full query)

---

## Next Steps

After completing this implementation:

1. **User Testing:** Get feedback from real users
2. **Performance Optimization:** Profile and optimize slow components
3. **Additional Personas:** Add more influencers to the system
4. **Advanced Features:**
   - Persona comparison tool
   - Multi-persona collaboration
   - Conversation history persistence
5. **Analytics:** Track usage patterns, popular queries, retrieval quality

---

## Conclusion

This implementation plan provides a comprehensive path to integrate your sophisticated influencer persona agent into Claude Code using modern 2025 best practices (Hybrid MCP + Skills).

The key innovation is the **two-tier linguistic style system**:
- **Tier 1 (Embedded):** Fast, reliable, always available
- **Tier 2 (Dynamic):** Complete, up-to-date, flexible

This ensures that personas remain authentic while benefiting from Claude Code's powerful environment.

**Estimated timeline:** 8-12 hours
**Risk level:** Low (incremental, testable phases)
**Maintenance:** Low (automated Skill generation)

Ready to begin implementation!
