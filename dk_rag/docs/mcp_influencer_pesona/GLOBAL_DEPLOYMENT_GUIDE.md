# Global MCP Deployment Guide: Influencer Persona Server

**Version**: 1.2 (MCP + Skills Global Deployment Complete)
**Last Updated**: 2025-10-30
**Platform**: macOS
**Claude Code Version**: Latest

**Status**: ✅ Fully Deployed and Working (MCP Server + Skills)

---

## 🎯 Quick Start (What Actually Works)

If you just want the working configuration without reading the full guide:

```json
{
  "mcpServers": {
    "persona-agent": {
      "command": "/opt/homebrew/bin/poetry",
      "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
```

**Key points**:
1. Use **absolute path** to Poetry (find with `which poetry`)
2. The `-C` flag is **critical** - it tells Poetry where your `pyproject.toml` is
3. Add this to `~/.claude.json` for global access
4. Test with `claude mcp list`

---

## Table of Contents

0. [🎯 Quick Start](#quick-start-what-actually-works)
1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Prerequisites](#prerequisites)
4. [Deployment Method 1: CLI (Recommended)](#deployment-method-1-cli-recommended)
5. [Deployment Method 2: Manual File Edit](#deployment-method-2-manual-file-edit)
6. [Verification & Testing](#verification--testing)
7. [Skills Global Deployment ✅](#skills-global-deployment--completed)
8. [Troubleshooting](#troubleshooting) ⭐ **See Poetry pyproject.toml issue**
9. [Maintenance & Updates](#maintenance--updates)
10. [Security Considerations](#security-considerations)
11. [Reference Documentation](#reference-documentation)

---

## Overview

This guide explains how to deploy your MCP influencer persona server **globally** on your Mac, making it available to **all Claude Code projects** without needing project-specific configuration.

### What You'll Achieve

- ✅ Access to 12 influencer personas from any Claude Code project
- ✅ MCP server globally configured via `~/.claude.json`
- ✅ Skills globally deployed to `~/.claude/skills/`
- ✅ Shell alias `claude-skills` for quick skill listing
- ✅ No need to copy configuration to each project
- ✅ Python MCP server scripts remain in their current location
- ✅ Skills available globally via `/dan-kennedy`, `/alex-hormozi`, etc.

### Current vs Global Setup

**Current (Project-Local)**:
```
/Users/blickt/Documents/src/pdf_2_text/.mcp.json
/Users/blickt/Documents/src/pdf_2_text/.claude/config.json
```
Only works when Claude Code is opened in the `pdf_2_text` directory.

**After Global Setup**:
```
~/.claude.json  (user-scoped configuration)
```
Works from **any directory** on your Mac.

---

## Key Concepts

### Configuration Scopes

Claude Code supports three configuration scopes with this priority order:

1. **local** (highest priority) - Temporary, current session only
2. **project** - Stored in `.mcp.json` (team-shared, version-controlled)
3. **user** (lowest priority) - Stored in `~/.claude.json` (your global config)

### File Locations on Mac

```
~/.claude.json                    # Global user MCP servers (THIS IS WHAT WE'LL CREATE)
~/.claude/settings.json           # Global user settings
.mcp.json                         # Project-level MCP servers
.claude/config.json              # Project-level config (alternative location)
```

### Why Absolute Paths?

Global configurations **must use absolute paths** because:
- Claude Code runs from different working directories
- `${workspaceFolder}` only works in project-level configs
- Absolute paths work consistently across all projects

---

## Prerequisites

### 1. Verify Python Installation

```bash
which python
python --version
```

Expected output: Path to Python (e.g., `/usr/bin/python3`) and version 3.8+

### 2. Verify MCP Server Exists

```bash
ls -l /Users/blickt/Documents/src/pdf_2_text/dk_rag/mcp_server/__main__.py
```

Expected: File should exist

### 3. Test MCP Server Locally

```bash
cd /Users/blickt/Documents/src/pdf_2_text
python -m dk_rag.mcp_server
```

Expected: Server should start without errors (may show MCP protocol messages)

Press `Ctrl+C` to stop.

### 4. Verify Claude Code CLI

```bash
claude --version
```

Expected: Claude Code version number

---

## Deployment Method 1: CLI (Recommended)

This is the **official and safest method** recommended by Anthropic.

### Step 1: Run the Add Command

**IMPORTANT**: Use the Poetry `-C` flag to specify the project directory:

```bash
claude mcp add-json persona-agent --scope user '{
  "command": "/opt/homebrew/bin/poetry",
  "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
  "env": {
    "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
  }
}'
```

### Understanding the Command

- `claude mcp add-json` - CLI command to add MCP server
- `persona-agent` - Server name (must match your current config)
- `--scope user` - **CRITICAL**: Makes it global (user-scoped)
- `/opt/homebrew/bin/poetry` - **Absolute path** to Poetry (use `which poetry` to find yours)
- `"-C", "/Users/blickt/Documents/src/pdf_2_text"` - **CRITICAL**: Tells Poetry which project directory to use
- `"run", "python", "-m", "dk_rag.mcp_server"` - Poetry runs Python from the virtualenv

### Step 2: Verify Installation

```bash
claude mcp list
```

Expected output:
```
persona-agent (user)
```

The `(user)` indicates it's globally configured.

### Step 3: Check Configuration File

```bash
cat ~/.claude.json
```

Expected: File should exist with your MCP server configuration.

---

## Deployment Method 2: Manual File Edit

Use this method if the CLI approach doesn't work or you prefer manual control.

### Step 1: Check if `~/.claude.json` Exists

```bash
ls -la ~/.claude.json
```

### Step 2A: If File Does NOT Exist

Create it with your MCP server configuration:

```bash
cat > ~/.claude.json << 'EOF'
{
  "mcpServers": {
    "persona-agent": {
      "command": "/opt/homebrew/bin/poetry",
      "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
EOF
```

**Note**: Replace `/opt/homebrew/bin/poetry` with your Poetry path (run `which poetry` to find it).

### Step 2B: If File Already Exists

**IMPORTANT**: Back up first!

```bash
# Create backup
cp ~/.claude.json ~/.claude.json.backup

# Edit the file
nano ~/.claude.json
```

Add your MCP server to the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "existing-server": {
      ...
    },
    "persona-agent": {
      "command": "/opt/homebrew/bin/poetry",
      "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
```

**Note**: Replace `/opt/homebrew/bin/poetry` with your Poetry path.

Save with `Ctrl+O`, exit with `Ctrl+X`.

### Step 3: Validate JSON Syntax

```bash
cat ~/.claude.json | python -m json.tool
```

Expected: Pretty-printed JSON without errors.

### Step 4: Verify with CLI

```bash
claude mcp list
```

Expected: `persona-agent (user)` should appear.

---

## Verification & Testing

### Test 1: Verify MCP Server List

```bash
claude mcp list
```

**Expected Output**:
```
persona-agent (user)
```

**Troubleshooting**: If not shown, see [Troubleshooting](#troubleshooting) section.

### Test 2: Test in Different Project Directory

```bash
# Navigate to a completely different directory
cd ~/Desktop
mkdir test-global-mcp
cd test-global-mcp

# Start Claude Code
claude
```

Inside Claude Code, check MCP status:
```
/mcp
```

**Expected**: Should show `persona-agent` as connected.

### Test 3: Test MCP Tools

In Claude Code, try using the MCP server:

```
Can you use the persona-agent MCP server to retrieve Dan Kennedy's mental models about direct response marketing?
```

**Expected**: Claude should successfully call `mcp__persona-agent__retrieve_mental_models` and return results.

### Test 4: Test Skills

```
/dan-kennedy
```

**Expected**: Dan Kennedy skill should activate with his greeting.

Ask a question:
```
How do I write a sales email that converts?
```

**Expected**: Response in Dan Kennedy's voice using the MCP tools.

### Test 5: Test in Original Project

```bash
cd /Users/blickt/Documents/src/pdf_2_text
claude
```

**Expected**: MCP server should still work (global config applies everywhere).

---

## Skills Global Deployment ✅ COMPLETED

**Status**: ✅ Deployed and Working
**Date Completed**: 2025-10-30
**Method Used**: Copy to Global Location (Option 1)

Skills have been successfully deployed globally and are now available in all projects on your Mac.

### What Are Claude Code Skills?

Skills are modular capabilities that extend Claude's functionality through organized folders containing instructions, scripts, and resources. They are automatically discovered by Claude from:

1. **Personal (Global) Skills**: `~/.claude/skills/` - Available across all projects
2. **Project Skills**: `.claude/skills/` - Shared with team via git
3. **Plugin Skills**: Installed via plugins

**Key Characteristics (2025 Best Practices)**:
- **Efficient**: Each skill uses only 30-50 tokens until loaded
- **Auto-discovery**: Claude automatically finds skills in standard locations
- **Model-invoked**: Claude decides when to use them based on task relevance
- **No registration needed**: Unlike MCP servers, skills are auto-discovered

### Skills Location

**Original (Project-level)**:
```
/Users/blickt/Documents/src/pdf_2_text/.claude/skills/
```

**Global (Deployed)**:
```
~/.claude/skills/
```

### Deployment Method Used: Copy to Global Location

This method creates independent copies of skills in the global directory, providing stability and independence from project changes.

```bash
# Create global skills directory
mkdir -p ~/.claude/skills

# Copy all 12 skills
cp -r /Users/blickt/Documents/src/pdf_2_text/.claude/skills/* ~/.claude/skills/

# Verify deployment
ls ~/.claude/skills/
```

**Result**: All 12 persona skills deployed globally.

### All 12 Global Skills Deployed

```
alex-hormozi        - Business scaling and acquisition strategies
ben-heath           - Facebook ads and paid marketing
copywrite-experts   - Copywriting and direct response
dan-kennedy         - Direct marketing and sales letters
dara-denney         - YouTube and content creation
greg-isenberg       - Startup ideas and validation
instantly-ai        - Cold email and outreach
koerner-office      - Hustle and business mindset
konstantinos        - E-commerce and dropshipping
mfm-hosts           - Entrepreneurship and trends (My First Million)
nick-theriot        - Agency growth and scaling
nihaixia            - Chinese fortune telling (Suanming)
```

### Alternative Deployment Options

**Option 2: Symlink Skills** (For active development)
```bash
# Symlink keeps single source of truth
mkdir -p ~/.claude/skills
ln -s /Users/blickt/Documents/src/pdf_2_text/.claude/skills/dan-kennedy ~/.claude/skills/dan-kennedy
# Repeat for other skills
```

**Pros**: Changes to project skills automatically reflect globally
**Cons**: Breaks if project directory moves

**Option 3: Regenerate Skills Script** (Future enhancement)
```bash
# Modify generation script to output directly to global location
# Update: dk_rag/scripts/generate_persona_skills.py
```

**Pros**: Integrated into workflow, clean regeneration
**Cons**: Requires code modification

### Verification & Testing

**1. List Global Skills**:
```bash
ls ~/.claude/skills/
```

Expected output: All 12 skill directories

**2. Use the Shell Alias** (Added to `~/.zshrc`):
```bash
claude-skills
```

Output:
```
=== Global Skills ===
alex-hormozi
ben-heath
copywrite-experts
dan-kennedy
dara-denney
greg-isenberg
instantly-ai
koerner-office
konstantinos
mfm-hosts
nick-theriot
nihaixia

=== Project Skills ===
(lists project skills if in a project directory)
```

**3. Test Skill Autocomplete**:
```bash
# In Claude Code, type and press Tab:
/alex-
# Should autocomplete to: /alex-hormozi
```

**4. Activate a Skill**:
```
/dan-kennedy
```

Expected: Greeting from Dan Kennedy in his authentic voice

**5. Test from Different Project**:
```bash
cd ~/Desktop
claude
/greg-isenberg
```

Expected: Skill works from any directory

**6. Ask Claude to List Skills**:
```
What skills are available?
```

Expected: Claude lists all 12 global skills

### How to List Available Skills

**No dedicated CLI command exists**, but you can use these methods:

**Method 1: Ask Claude** (Recommended)
```
What skills are available?
```

**Method 2: Filesystem**
```bash
ls ~/.claude/skills/          # Global skills
ls .claude/skills/            # Project skills
```

**Method 3: Shell Alias** (Now configured)
```bash
claude-skills
```

**Method 4: Autocomplete**
```
# Type / in Claude Code and press Tab
/
```

### Shell Alias Configuration

Added to `~/.zshrc` for quick skill listing:

```bash
# Claude Code Skills - Quick list command
alias claude-skills='echo "=== Global Skills ===" && ls ~/.claude/skills/ && echo "\n=== Project Skills ===" && ls .claude/skills/ 2>/dev/null || echo "(none)"'
```

**Usage**: Open a new terminal or run `source ~/.zshrc`, then use `claude-skills`

---

## Troubleshooting

### Problem: Server Not Appearing in `claude mcp list`

**Diagnosis**:
```bash
# Check if file exists
ls -la ~/.claude.json

# Validate JSON syntax
cat ~/.claude.json | python -m json.tool
```

**Solutions**:
1. Verify file is valid JSON
2. Restart Claude Code completely
3. Try removing and re-adding: `claude mcp remove persona-agent && claude mcp add-json ...`

### Problem: "spawn ENOENT" or Command Not Found

**Error Message**: `Error: spawn python ENOENT`

**Solutions**:

1. Use absolute Python path:
   ```bash
   which python  # Get full path, e.g., /usr/bin/python3
   ```

   Update config:
   ```json
   {
     "command": "/usr/bin/python3",
     "args": ["-m", "dk_rag.mcp_server"],
     "env": {
       "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
     }
   }
   ```

2. Verify Python is in PATH:
   ```bash
   echo $PATH
   ```

### Problem: Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'dk_rag'`

**Solutions**:

1. Verify PYTHONPATH is correct:
   ```bash
   ls /Users/blickt/Documents/src/pdf_2_text/dk_rag
   ```

2. Test module import manually:
   ```bash
   cd /Users/blickt/Documents/src/pdf_2_text
   python -c "import dk_rag.mcp_server"
   ```

3. Check `__main__.py` exists:
   ```bash
   ls -l /Users/blickt/Documents/src/pdf_2_text/dk_rag/mcp_server/__main__.py
   ```

### Problem: Poetry "Could not find pyproject.toml" Error

**Error**: `Poetry could not find a pyproject.toml file in /Users/blickt or its parents`

**Cause**: When Claude Code spawns the MCP server in a clean environment, Poetry doesn't know which project directory to use.

**Solutions**:

**✅ RECOMMENDED: Use Poetry's `-C` flag** (This is what works!):
```json
{
  "command": "/opt/homebrew/bin/poetry",
  "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
  "env": {
    "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
  }
}
```

**Key points**:
1. Use **absolute path** to Poetry: `/opt/homebrew/bin/poetry` (find yours with `which poetry`)
2. Add `"-C", "/path/to/project"` **before** `"run"` in the args array
3. This tells Poetry which directory contains `pyproject.toml`
4. The `cwd` field is **not sufficient** - you must use `-C`

**❌ What DOESN'T work**:
```json
// WRONG: Just using "poetry" without absolute path
{"command": "poetry", ...}

// WRONG: Using cwd without -C flag
{"command": "/opt/homebrew/bin/poetry", "cwd": "/path/to/project", "args": ["run", ...]}

// WRONG: Using plain python (missing dependencies)
{"command": "python", "args": ["-m", "dk_rag.mcp_server"]}
```

**Why this happens**:
- Claude Code spawns MCP servers in a clean shell environment
- Poetry needs to know the project root to find `pyproject.toml` and the virtualenv
- The `cwd` configuration field doesn't always set the working directory before Poetry runs
- Using `-C` explicitly tells Poetry where to look

### Problem: Server Shows "Failed" Status

**Diagnosis**:
```bash
# Run with debug mode
claude --mcp-debug

# Or test server directly
cd /Users/blickt/Documents/src/pdf_2_text
python -m dk_rag.mcp_server
```

**Solutions**:
1. Check server logs for errors
2. Verify all dependencies are installed
3. Test MCP server independently using MCP Inspector:
   ```bash
   npx @modelcontextprotocol/inspector python -m dk_rag.mcp_server
   ```

### Problem: Tools Not Working But Server Connected

**Symptoms**: MCP server shows as connected but tools fail when called.

**Solutions**:
1. Verify persona data is accessible:
   ```bash
   ls /Volumes/J15/aicallgo_data/persona_data_base/personas/
   ```

2. Check MCP tool names:
   - Should be `mcp__persona-agent__retrieve_mental_models`
   - Not `retrieve_mental_models` (missing prefix)

3. Test tools manually via MCP Inspector

### Problem: Skills Not Available Globally

**Solutions**:
1. Verify skills copied to `~/.claude/skills/`:
   ```bash
   ls ~/.claude/skills/
   ```

2. Restart Claude Code after copying skills

3. Check skill autocomplete:
   - Type `/dan-` and press Tab
   - Should show `/dan-kennedy`

---

## Maintenance & Updates

### Updating Persona Data

When you update persona data or regenerate artifacts:

1. **Regenerate Skills** in project directory:
   ```bash
   cd /Users/blickt/Documents/src/pdf_2_text
   make generate-skills
   ```

2. **Update Global Skills** (✅ REQUIRED - global skills are now deployed):
   ```bash
   # Sync project skills to global location
   cp -r /Users/blickt/Documents/src/pdf_2_text/.claude/skills/* ~/.claude/skills/
   ```

3. **Verify the update**:
   ```bash
   claude-skills
   # Or: ls ~/.claude/skills/
   ```

4. **Restart Claude Code** to reload skills

**Note**: Since we deployed using the copy method, you must manually sync project skills to global location after regeneration. For automatic sync, consider using symlinks instead (see Alternative Deployment Options).

### Updating MCP Server Code

When you modify the MCP server Python code:

1. **No config changes needed** - global config points to the code location
2. **Restart Claude Code** to reload the server
3. **Test in any project** to verify changes work

### Updating Configuration

To modify the global MCP configuration:

```bash
# Method 1: Edit file directly
nano ~/.claude.json

# Method 2: Remove and re-add
claude mcp remove persona-agent
claude mcp add-json persona-agent --scope user '{...}'
```

### Viewing Current Configuration

**MCP Server Configuration**:
```bash
# List all MCP servers
claude mcp list

# Get specific server config
claude mcp get persona-agent

# View full config file
cat ~/.claude.json
```

**Skills Configuration**:
```bash
# List all global skills
claude-skills

# Or manually:
ls ~/.claude/skills/

# View a specific skill
cat ~/.claude/skills/dan-kennedy/SKILL.md

# Count skills
ls ~/.claude/skills/ | wc -l
```

### Removing Global Configuration

If you need to remove the global MCP server:

```bash
# Method 1: CLI
claude mcp remove persona-agent

# Method 2: Manual
nano ~/.claude.json
# Delete the "persona-agent" entry

# Verify removal
claude mcp list
```

---

## Security Considerations

### Auto-Approval Behavior

**Important**: User-scoped MCP servers in `~/.claude.json` may **auto-approve tool calls** without manual confirmation.

**Implications**:
- Faster workflow (no approval prompts)
- Less manual oversight
- Trust is critical

**Best Practice**: Only deploy MCP servers you fully trust to global scope.

### File System Access

Your MCP server has access to:
- `/Users/blickt/Documents/src/pdf_2_text/` (via PYTHONPATH)
- `/Volumes/J15/aicallgo_data/persona_data_base/` (persona data)

**Review**: Check `dk_rag/mcp_server/persona_mcp_server.py` for:
- File read operations
- Data access patterns
- No unintended file writes

### Absolute Paths Security

**Consideration**: Absolute paths in global config are machine-specific.

**Implications**:
- Config won't work if shared with others
- Paths won't work if you move project directory
- Others can't use your global config directly

**Mitigation**: Document paths clearly; use relative paths in project-level configs for team sharing.

### Configuration Hierarchy

When the same server name exists at multiple scopes:

1. **local** scope - overrides everything (temporary)
2. **project** scope - overrides user scope (team)
3. **user** scope - global default (you)

**Implication**: If a project has `.mcp.json` with `persona-agent`, it will override your global config in that project.

### Version Control

**Rules**:
- `~/.claude.json` → **NEVER commit** (machine-specific)
- `.mcp.json` → **DO commit** (team-shared)
- `.claude/settings.local.json` → **NEVER commit** (in `.gitignore`)

Check your `.gitignore`:
```bash
cat /Users/blickt/Documents/src/pdf_2_text/.gitignore | grep claude
```

Should include:
```
.claude/settings.local.json
```

---

## Reference Documentation

### Official Anthropic Documentation

1. **Claude Code MCP Guide**
   https://docs.claude.com/en/docs/claude-code/mcp
   - Configuration files and scopes
   - Environment variables
   - CLI commands

2. **Claude Code Settings**
   https://docs.claude.com/en/docs/claude-code/settings
   - Configuration hierarchy
   - File locations on different platforms

3. **Model Context Protocol**
   https://modelcontextprotocol.io/docs/develop/connect-local-servers
   - MCP server connection patterns
   - Protocol specifications

### Community Resources

4. **Scott Spence - Configuring MCP Tools**
   https://scottspence.com/posts/configuring-mcp-tools-in-claude-code
   - Practical examples
   - Best practices

5. **CloudArtisan - Adding MCP Servers**
   https://cloudartisan.com/posts/2025-04-12-adding-mcp-servers-claude-code/
   - CLI command examples
   - Troubleshooting tips

6. **MCPcat Setup Guide**
   https://mcpcat.io/guides/adding-an-mcp-server-to-claude-code/
   - Visual step-by-step guide

### GitHub Issues (Resolved)

7. **Issue #515 - Global MCP Servers**
   https://github.com/anthropics/claude-code/issues/515
   - Clarified correct file location
   - Status: COMPLETED

8. **Issue #4976 - Documentation Corrections**
   https://github.com/anthropics/claude-code/issues/4976
   - Fixed incorrect config file paths

### Related Documentation

- **Your Implementation**: `/Users/blickt/Documents/src/pdf_2_text/dk_rag/docs/mcp_influencer_pesona/IMPLEMENTATION_COMPLETE.md`
- **MCP Server README**: `/Users/blickt/Documents/src/pdf_2_text/dk_rag/mcp_server/README.md`

---

## Quick Reference

### Essential Commands

```bash
# Add server globally (use absolute path to poetry and -C flag)
claude mcp add-json persona-agent --scope user '{
  "command": "/opt/homebrew/bin/poetry",
  "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
  "env": {
    "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
  }
}'

# List all MCP servers
claude mcp list

# Get server config
claude mcp get persona-agent

# Remove server
claude mcp remove persona-agent

# Run with debug
claude --mcp-debug

# Test server directly
cd /Users/blickt/Documents/src/pdf_2_text
python -m dk_rag.mcp_server

# Validate config file
cat ~/.claude.json | python -m json.tool
```

### Configuration Template

**✅ WORKING CONFIGURATION** (Tested and verified):

```json
{
  "mcpServers": {
    "persona-agent": {
      "command": "/opt/homebrew/bin/poetry",
      "args": ["-C", "/Users/blickt/Documents/src/pdf_2_text", "run", "python", "-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
```

**Important notes**:
- Replace `/opt/homebrew/bin/poetry` with your Poetry path (run `which poetry`)
- The `-C` flag is **required** to tell Poetry where `pyproject.toml` is located
- Use absolute paths for both the command and the project directory

### File Locations

```
~/.claude.json                                    # Global MCP config (CREATE THIS)
~/.claude/skills/                                 # Global skills (optional)
/Users/blickt/Documents/src/pdf_2_text/dk_rag/mcp_server/    # MCP server code
/Volumes/J15/aicallgo_data/persona_data_base/     # Persona data
```

---

## Success Checklist

After completing this guide, verify:

**MCP Server Deployment**:
- [x] `claude mcp list` shows `persona-agent (user)`
- [x] MCP server works from `~/Desktop` test directory
- [x] MCP tools can be called successfully
- [x] `~/.claude.json` exists and is valid JSON
- [x] No errors in Claude Code MCP logs

**Skills Deployment**:
- [x] `~/.claude/skills/` directory exists with 12 skills
- [x] `claude-skills` alias works in terminal
- [x] Skills autocomplete works: `/dan-` → `/dan-kennedy`
- [x] Skills activate successfully: `/alex-hormozi` shows greeting
- [x] Skills work from any project directory

**Integration Testing**:
- [x] Personas respond in authentic voice
- [x] MCP tools + Skills work together seamlessly
- [x] Original project still works correctly
- [x] Shell alias added to `~/.zshrc`

---

## Support & Issues

If you encounter issues not covered in this guide:

1. **Check Claude Code logs**: `~/.claude/logs/mcp/`
2. **Test server independently**: Use MCP Inspector
3. **Search GitHub issues**: https://github.com/anthropics/claude-code/issues
4. **Review MCP protocol docs**: https://modelcontextprotocol.io
5. **Check your implementation docs**: `IMPLEMENTATION_COMPLETE.md`

---

## 🎉 Summary: What We Accomplished

This guide documented the complete global deployment of the influencer persona system on macOS. Here's what was achieved:

### Phase 1: MCP Server Global Deployment (Previously Completed)
- ✅ Configured `~/.claude.json` with persona-agent MCP server
- ✅ Used Poetry with `-C` flag for proper project directory resolution
- ✅ Verified MCP server works from any project on the Mac
- ✅ All 3 MCP tools accessible: `retrieve_mental_models`, `retrieve_core_beliefs`, `retrieve_transcripts`

### Phase 2: Skills Global Deployment (Completed 2025-10-30)
- ✅ Created `~/.claude/skills/` directory
- ✅ Deployed all 12 influencer persona skills globally
- ✅ Added `claude-skills` shell alias to `~/.zshrc`
- ✅ Verified skills work from any project directory
- ✅ Documented best practices from Claude Code 2025 guidelines

### Global Skills Deployed
All 12 persona skills are now available system-wide:
1. **alex-hormozi** - Business scaling and acquisition
2. **ben-heath** - Facebook ads and paid marketing
3. **copywrite-experts** - Copywriting and direct response
4. **dan-kennedy** - Direct marketing and sales letters
5. **dara-denney** - YouTube and content creation
6. **greg-isenberg** - Startup ideas and validation
7. **instantly-ai** - Cold email and outreach
8. **koerner-office** - Hustle and business mindset
9. **konstantinos** - E-commerce and dropshipping
10. **mfm-hosts** - Entrepreneurship (My First Million)
11. **nick-theriot** - Agency growth and scaling
12. **nihaixia** - Chinese fortune telling (Suanming)

### Developer Experience Improvements
- **Shell Alias**: `claude-skills` command for quick skill listing
- **Auto-discovery**: Skills automatically found by Claude, no manual registration
- **Efficient Loading**: Only 30-50 tokens per skill until activated
- **Portable**: Works from any project directory on the Mac

### Key Learnings & Best Practices (2025)
1. **MCP vs Skills**: MCP servers require registration in `~/.claude.json`, skills are auto-discovered
2. **Poetry Flag**: Use `-C` flag to specify project directory for global deployments
3. **Copy Method**: Provides stability; requires manual sync after skill regeneration
4. **No CLI Command**: No `claude skills list` command exists; use filesystem or ask Claude
5. **Auto-approval**: User-scoped MCP servers auto-approve tool calls for faster workflow

### Files Modified/Created
```
~/.claude.json           (MCP server config)
~/.claude/skills/        (12 skill directories)
~/.zshrc                 (shell alias added)
GLOBAL_DEPLOYMENT_GUIDE.md (this document - updated)
```

### Next Steps & Maintenance
When updating persona data:
1. Regenerate skills: `make generate-skills`
2. Sync to global: `cp -r .claude/skills/* ~/.claude/skills/`
3. Verify: `claude-skills`
4. Restart Claude Code

**Future Enhancement**: Consider updating `generate_persona_skills.py` to output directly to `~/.claude/skills/` for automated workflow.

---

**Last Updated**: 2025-10-30
**Author**: Generated by Claude Code
**Version**: 1.2
**Status**: Production Ready - MCP Server + Skills Fully Deployed
