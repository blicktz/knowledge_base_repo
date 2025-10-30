# Global MCP Deployment Guide: Influencer Persona Server

**Version**: 1.0
**Last Updated**: 2025-10-30
**Platform**: macOS
**Claude Code Version**: Latest

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Prerequisites](#prerequisites)
4. [Deployment Method 1: CLI (Recommended)](#deployment-method-1-cli-recommended)
5. [Deployment Method 2: Manual File Edit](#deployment-method-2-manual-file-edit)
6. [Verification & Testing](#verification--testing)
7. [Skills Global Deployment (Optional)](#skills-global-deployment-optional)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance & Updates](#maintenance--updates)
10. [Security Considerations](#security-considerations)
11. [Reference Documentation](#reference-documentation)

---

## Overview

This guide explains how to deploy your MCP influencer persona server **globally** on your Mac, making it available to **all Claude Code projects** without needing project-specific configuration.

### What You'll Achieve

- Access to 12 influencer personas from any Claude Code project
- No need to copy configuration to each project
- Python MCP server scripts remain in their current location
- Skills available globally via `/dan-kennedy`, `/alex-hormozi`, etc.

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

```bash
claude mcp add-json persona-agent --scope user '{
  "command": "python",
  "args": ["-m", "dk_rag.mcp_server"],
  "env": {
    "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
  }
}'
```

### Understanding the Command

- `claude mcp add-json` - CLI command to add MCP server
- `persona-agent` - Server name (must match your current config)
- `--scope user` - **CRITICAL**: Makes it global (user-scoped)
- JSON config - Same as your current `.mcp.json` config

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
      "command": "python",
      "args": ["-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
EOF
```

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
      "command": "python",
      "args": ["-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
```

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

## Skills Global Deployment (Optional)

Skills can also be deployed globally so they're available in all projects.

### Current Skills Location

```
/Users/blickt/Documents/src/pdf_2_text/.claude/skills/
```

### Option 1: Copy Skills to Global Location

```bash
# Create global skills directory
mkdir -p ~/.claude/skills

# Copy all skills
cp -r /Users/blickt/Documents/src/pdf_2_text/.claude/skills/* ~/.claude/skills/
```

### Option 2: Symlink Skills

```bash
# Create global skills directory
mkdir -p ~/.claude/skills

# Symlink each skill
ln -s /Users/blickt/Documents/src/pdf_2_text/.claude/skills/dan-kennedy ~/.claude/skills/dan-kennedy
ln -s /Users/blickt/Documents/src/pdf_2_text/.claude/skills/alex-hormozi ~/.claude/skills/alex-hormozi
# ... repeat for other skills
```

### Option 3: Regenerate Skills Script

Update `dk_rag/scripts/generate_persona_skills.py` to output to `~/.claude/skills/` instead.

### Verify Global Skills

```bash
ls -l ~/.claude/skills/
```

Expected: All 12 persona skill directories.

Test in any project:
```
claude
/dan-kennedy
```

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

1. **Regenerate Skills** (if skills are in project):
   ```bash
   cd /Users/blickt/Documents/src/pdf_2_text
   make generate-skills
   ```

2. **Update Global Skills** (if using global skills):
   ```bash
   cp -r /Users/blickt/Documents/src/pdf_2_text/.claude/skills/* ~/.claude/skills/
   ```

3. **Restart Claude Code** to reload skills

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

```bash
# List all MCP servers
claude mcp list

# Get specific server config
claude mcp get persona-agent

# View full config file
cat ~/.claude.json
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
# Add server globally
claude mcp add-json persona-agent --scope user '{
  "command": "python",
  "args": ["-m", "dk_rag.mcp_server"],
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

```json
{
  "mcpServers": {
    "persona-agent": {
      "command": "python",
      "args": ["-m", "dk_rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/blickt/Documents/src/pdf_2_text"
      }
    }
  }
}
```

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

- [ ] `claude mcp list` shows `persona-agent (user)`
- [ ] MCP server works from `~/Desktop` test directory
- [ ] Skills autocomplete works: `/dan-` → `/dan-kennedy`
- [ ] MCP tools can be called successfully
- [ ] Personas respond in authentic voice
- [ ] No errors in Claude Code MCP logs
- [ ] `~/.claude.json` exists and is valid JSON
- [ ] Original project still works correctly

---

## Support & Issues

If you encounter issues not covered in this guide:

1. **Check Claude Code logs**: `~/.claude/logs/mcp/`
2. **Test server independently**: Use MCP Inspector
3. **Search GitHub issues**: https://github.com/anthropics/claude-code/issues
4. **Review MCP protocol docs**: https://modelcontextprotocol.io
5. **Check your implementation docs**: `IMPLEMENTATION_COMPLETE.md`

---

**Last Updated**: 2025-10-30
**Author**: Generated by Claude Code
**Version**: 1.0
**Status**: Production Ready
