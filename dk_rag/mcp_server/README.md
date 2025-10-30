# Persona Agent MCP Server

This directory contains the Model Context Protocol (MCP) server for the Influencer Persona Agent. The server provides 3 data retrieval tools that integrate with Claude Code Skills to enable authentic influencer persona interactions.

## Architecture

This is a **simplified architecture** following 2025 best practices:

- **MCP Server = Data Only**: Provides 3 tools for retrieving knowledge
- **Skills = Everything Else**: Full linguistic style, query analysis, and workflows embedded in Skill markdown files

## Components

### MCP Server (`persona_mcp_server.py`)

Provides 3 data retrieval tools:

1. **`retrieve_mental_models`** - Step-by-step frameworks and mental models
   - Use for: "how-to" questions that need process guidance
   - Returns: Structured frameworks with name, description, and steps

2. **`retrieve_core_beliefs`** - Philosophical principles and core beliefs
   - Use for: "why" questions and opinion-based queries
   - Returns: Belief statements with category and supporting evidence

3. **`retrieve_transcripts`** - Real examples, stories, and anecdotes
   - Use for: Factual queries and concrete evidence
   - Returns: Transcript chunks with content, metadata, and relevance scores

### Skills Generator (`../scripts/generate_persona_skills.py`)

Auto-generates Claude Code Skills from persona artifacts. Each Skill contains:
- Complete linguistic style profile (tone, catchphrases, vocabulary, sentence patterns)
- Query analysis instructions
- Language detection rules
- Tool calling workflow logic

## Setup

### 1. Install Dependencies

The MCP package is already included in the project dependencies:

```bash
poetry install
```

### 2. Configure Claude Code

The MCP server is configured in `.claude/config.json`:

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

### 3. Generate Skills

Generate Skills for all personas:

```bash
make generate-skills
```

Or for a specific persona:

```bash
make generate-skill-dan_kennedy_god_of_direct_response_marketing
```

Skills are generated in `.claude/skills/{persona_id}/SKILL.md`

## Usage

### In Claude Code

1. **Activate a Persona Skill:**
   ```
   /dan-kennedy
   ```

2. **Ask Questions:**
   ```
   How do I write a sales email that converts?
   ```

3. **Behind the Scenes:**
   - Claude analyzes intent (instructional_inquiry)
   - Calls `retrieve_mental_models` for frameworks
   - Calls `retrieve_transcripts` for examples
   - Responds in Dan Kennedy's voice with catchphrases

### Available Personas

List all available personas:
```bash
ls .claude/skills/
```

Current personas and their slash commands:
- `/dan-kennedy` - Dan Kennedy (Direct Response Marketing)
- `/alex-hormozi` - Alex Hormozi (Business Growth)
- `/greg-isenberg` - Greg Isenberg (Startup Strategies)
- `/mfm-hosts` - My First Million Hosts
- `/ben-heath` - Ben Heath
- `/nick-theriot` - Nick Theriot
- `/dara-denney` - Dara Denney
- `/koerner-office` - Koerner Office
- `/copywrite-experts` - Copywrite Experts
- `/instantly-ai` - Instantly AI Founders
- `/konstantinos` - Konstantinos Doulgeridis
- `/nihaixia` - Nihaixia Suanming

## Testing

### Test MCP Server Standalone

```bash
make test-mcp-server
```

This starts the MCP server and waits for stdin/stdout communication.

### Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python -m dk_rag.mcp_server
```

Opens a web UI to test the MCP tools interactively.

## Development

### Adding New Personas

1. **Create persona data** (Phase 1 extraction pipeline)
2. **Add slug mapping** in `generate_persona_skills.py` slug_map
3. **Generate Skill:**
   ```bash
   make generate-skill-new_persona_id
   ```
4. **Test in Claude Code:**
   ```
   /new-slug
   ```

### Updating Persona Data

When persona artifacts are updated:

1. **Regenerate Skills:**
   ```bash
   make generate-skills
   ```

2. **Restart Claude Code** (to reload Skills)

3. **Test updated persona**

## Troubleshooting

### Skills not appearing

- Check Skills exist in `.claude/skills/`
- Verify each SKILL.md has YAML frontmatter
- Restart Claude Code

### MCP tools not available

- Run: `claude mcp list` (if available)
- Check logs: `~/.claude/logs/mcp/persona-agent.log`
- Test server: `make test-mcp-server`

### Response not in persona's voice

- Regenerate Skills: `make generate-skills`
- Check artifact has complete `linguistic_style` section
- Verify Skill file contains full style profile

### Wrong language

- Ensure query is clearly in target language
- Check Skill has language detection instructions
- Skills should respond entirely in the detected language

### No retrieval results

- Verify persona_id is correct
- Check vector DB exists for persona
- Test KnowledgeIndexer methods directly

## File Structure

```
dk_rag/mcp_server/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point (python -m dk_rag.mcp_server)
├── persona_mcp_server.py    # Main MCP server (3 tools only!)
└── README.md                # This file

.claude/
├── config.json              # MCP server configuration
└── skills/                  # Generated Skills
    ├── dan-kennedy/
    │   └── SKILL.md         # Full style embedded!
    ├── alex-hormozi/
    │   └── SKILL.md
    └── ...
```

## Implementation Details

### Lazy Loading

The server uses lazy loading for persona-specific components:
- KnowledgeIndexer instances are cached per persona
- Retrieval pipelines are initialized on first use
- This reduces startup time and memory usage

### Error Handling

All tool calls include try/catch blocks and return:
- On success: JSON with `tool`, `persona_id`, `query`, `results`, `count`
- On error: JSON with `error`, `tool`, `persona_id`, `query`

### Language Support

The MCP server is language-agnostic. Language handling is managed by:
- Skills (detect language, enforce output language)
- KnowledgeIndexer (handles multilingual data)

Currently supported languages:
- English (en)
- Chinese (zh)

## Performance

Typical response times:
- Mental Models: ~500ms
- Core Beliefs: ~500ms
- Transcripts: ~1-2s (includes HyDE + hybrid + reranking)

Total workflow: ~2-4 seconds for a complete query

## License

See project root LICENSE file.
