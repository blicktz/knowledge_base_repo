# Adding New Personas as Claude Skills

This guide provides a complete step-by-step process for converting a persona into a Claude Code skill.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Step-by-Step Process](#step-by-step-process)
- [Understanding the Structure](#understanding-the-structure)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## Prerequisites

Before adding a new persona as a skill, ensure you have:

1. **Persona artifact created** in `/Volumes/J15/aicallgo_data/persona_data_base/personas/<persona_id>/artifacts/`
   - Must be a `.json.gz` compressed file
   - Contains linguistic style profile (tone, catchphrases, vocabulary, etc.)

2. **Persona registered** in `/Volumes/J15/aicallgo_data/persona_data_base/personas/persona_registry.json`
   - Entry includes: name, id, created_at, metadata, stats, latest_artifact

3. **Vector databases created** (optional but recommended):
   - `vector_db_mental_models/` - ChromaDB for mental models
   - `vector_db_core_beliefs/` - ChromaDB for core beliefs
   - `vector_db/` - General vector database for transcripts

4. **Poetry environment** set up for the project

---

## Overview

The process involves:
1. Adding a slug mapping for the new persona
2. Running the automated skill generator
3. Verifying the skill was created successfully

**Time Required:** ~5 minutes per persona

**Automated vs Manual:** We use the automated script for consistency and accuracy.

---

## Step-by-Step Process

### Step 1: Add Slug Mapping

The slug is the short command name users will type to activate the skill (e.g., `/alex-hormozi`, `/starter-story`).

**File to Edit:** `/Users/blickt/Documents/src/pdf_2_text/dk_rag/scripts/generate_persona_skills.py`

**Location:** Lines 86-100 (in the `_create_skill_slug()` method)

**Action:**
1. Open the file
2. Find the `slug_map` dictionary
3. Add a new entry mapping your persona ID to a user-friendly slug

**Example:**
```python
slug_map = {
    'dan_kennedy_god_of_direct_response_marketing': 'dan-kennedy',
    'alex_hormozi_youtuber': 'alex-hormozi',
    'my_first_million_show_hosts': 'mfm-hosts',
    'starter_story_youtuber': 'starter-story',  # ← Add your new entry here
    'your_new_persona_id': 'your-slug',         # ← Format: lowercase, hyphenated
}
```

**Slug Naming Guidelines:**
- Use lowercase letters only
- Separate words with hyphens (-)
- Keep it short and memorable (2-3 words max)
- Match the persona's brand/name
- Avoid underscores or special characters

### Step 2: Run the Skill Generator

**Command:**
```bash
cd /Users/blickt/Documents/src/pdf_2_text

poetry run python -m dk_rag.scripts.generate_persona_skills \
  --persona-id <your_persona_id> \
  --output-dir /Users/blickt/.claude/skills
```

**Parameters:**
- `--persona-id`: The exact persona ID from persona_registry.json
- `--output-dir`: Where to save the skill (default: `.claude/skills`)

**What Happens:**
1. Script loads the latest persona artifact from disk
2. Extracts linguistic style profile:
   - Tone
   - Catchphrases
   - Vocabulary
   - Sentence structures
   - Communication style
3. Generates a complete SKILL.md file using the template
4. Saves to `/Users/blickt/.claude/skills/<slug>/SKILL.md`

**Expected Output:**
```
11:21:36 - INFO - [ArtifactDisc] Found 1 artifacts for persona 'your_persona_id'
11:21:36 - INFO - [ArtifactDisc] Latest artifact for 'your_persona_id': persona_your_persona_id_20251108_065927.json.gz (timestamp: 2025-11-08 06:59:27)
✓ Generated Skill for: your_persona_id
```

### Step 3: Verify Skill Creation

**Check 1: File exists**
```bash
ls -la /Users/blickt/.claude/skills/<your-slug>/
```

Expected output:
```
total 24
drwxr-xr-x@  3 blickt  staff    96 Nov  8 11:21 .
drwxr-xr-x@ 15 blickt  staff   480 Nov  8 11:21 ..
-rw-r--r--@  1 blickt  staff  XXXX Nov  8 11:21 SKILL.md
```

**Check 2: Review the file content**
```bash
head -20 /Users/blickt/.claude/skills/<your-slug>/SKILL.md
```

Verify it contains:
- YAML frontmatter with name and description
- Linguistic style profile
- Catchphrases
- Vocabulary
- Workflow instructions

**Check 3: Test in Claude Code**

1. Restart Claude Code (if running)
2. Type `/<your-slug>` in the chat
3. The skill should activate
4. You should see a greeting in the persona's voice

---

## Understanding the Structure

### Persona Artifact Structure

The persona artifact (`.json.gz`) contains:

```json
{
  "persona_id": "starter_story_youtuber",
  "created_at": "2025-11-08T06:59:27",
  "linguistic_style": {
    "tone": "Energetic, conversational...",
    "catchphrases": ["I'm Pat Walls...", "Let's dive in..."],
    "vocabulary": ["MRR", "ARR", "productized service..."],
    "sentence_structures": ["Short punchy declaratives...", "..."],
    "communication_style": {
      "formality": "very_informal",
      "directness": "very_direct",
      "use_of_examples": "constant",
      "storytelling": "constant",
      "humor": "occasional"
    }
  },
  "mental_models": [...],
  "core_beliefs": [...],
  "sample_responses": [...]
}
```

### Generated Skill Structure

The generated `SKILL.md` file has these sections:

1. **YAML Frontmatter** - Name and description for Claude Code UI
2. **Linguistic Style Profile** - Complete voice/tone guidelines
3. **Initialization** - How to greet users when skill activates
4. **Query Processing Workflow** - 5-step process for handling queries
5. **MCP Tools Available** - Three retrieval tools for mental models, core beliefs, and transcripts
6. **Final Response Requirements** - Quality checklist

### File Locations Reference

```
Project Structure:
/Users/blickt/Documents/src/pdf_2_text/
├── dk_rag/
│   └── scripts/
│       └── generate_persona_skills.py    # Skill generator script
│
/Users/blickt/.claude/skills/              # Claude Code skills directory
├── alex-hormozi/
│   └── SKILL.md
├── starter-story/
│   └── SKILL.md
└── your-slug/
    └── SKILL.md
│
/Volumes/J15/aicallgo_data/persona_data_base/personas/
├── persona_registry.json                  # Master persona registry
├── alex_hormozi_youtuber/
│   ├── artifacts/
│   │   └── persona_alex_hormozi_youtuber_*.json.gz
│   ├── vector_db_mental_models/
│   ├── vector_db_core_beliefs/
│   └── vector_db/
└── your_new_persona/
    ├── artifacts/
    ├── vector_db_mental_models/
    ├── vector_db_core_beliefs/
    └── vector_db/
```

---

## Verification

### Complete Verification Checklist

- [ ] Slug mapping added to `generate_persona_skills.py`
- [ ] Script ran without errors
- [ ] Directory created: `/Users/blickt/.claude/skills/<slug>/`
- [ ] File exists: `SKILL.md` (typically 8-10 KB)
- [ ] YAML frontmatter present with correct name and description
- [ ] Linguistic style section populated with:
  - [ ] Tone description
  - [ ] Catchphrases list (at least 5+)
  - [ ] Vocabulary list (at least 10+ terms)
  - [ ] Sentence structure patterns
  - [ ] Communication style requirements
- [ ] Workflow section includes all 5 steps
- [ ] MCP tools section references correct `persona_id`
- [ ] Skill activates in Claude Code with `/<slug>`
- [ ] Greeting message appears in persona's voice

### Quick Test Commands

```bash
# Check file size (should be ~8-10 KB)
ls -lh /Users/blickt/.claude/skills/<slug>/SKILL.md

# Check YAML frontmatter
head -5 /Users/blickt/.claude/skills/<slug>/SKILL.md

# Check for catchphrases
grep -A 10 "Catchphrases" /Users/blickt/.claude/skills/<slug>/SKILL.md

# Check persona_id is correct throughout
grep "persona_id" /Users/blickt/.claude/skills/<slug>/SKILL.md
```

---

## Troubleshooting

### Common Issues

#### 1. **ModuleNotFoundError: No module named 'yaml'**

**Problem:** Running script with `python3` instead of `poetry run python`

**Solution:**
```bash
# ❌ Wrong
python3 -m dk_rag.scripts.generate_persona_skills ...

# ✅ Correct
poetry run python -m dk_rag.scripts.generate_persona_skills ...
```

#### 2. **No artifacts found for persona**

**Problem:** Persona artifact doesn't exist or is in wrong location

**Solution:**
```bash
# Check if artifact exists
ls -la /Volumes/J15/aicallgo_data/persona_data_base/personas/<persona_id>/artifacts/

# Verify persona is in registry
cat /Volumes/J15/aicallgo_data/persona_data_base/personas/persona_registry.json | grep "<persona_id>"
```

#### 3. **Skill not appearing in Claude Code**

**Problem:** Claude Code hasn't refreshed or file permissions issue

**Solution:**
1. Restart Claude Code
2. Check file permissions: `chmod 644 /Users/blickt/.claude/skills/<slug>/SKILL.md`
3. Verify directory permissions: `chmod 755 /Users/blickt/.claude/skills/<slug>/`

#### 4. **Empty or missing linguistic style sections**

**Problem:** Persona artifact doesn't contain linguistic_style data

**Solution:**
1. Verify artifact has linguistic_style: `gunzip -c <artifact.json.gz> | grep "linguistic_style"`
2. If missing, regenerate persona artifact with linguistic style analysis
3. Ensure persona pipeline includes linguistic profiling step

#### 5. **Wrong persona_id in generated skill**

**Problem:** Slug mapping or template substitution error

**Solution:**
1. Check slug_map entry matches persona_id exactly
2. Re-run generator with correct `--persona-id` parameter
3. Verify persona_registry.json has correct ID

---

## Examples

### Example 1: Adding "Dan Kennedy" Persona

**Persona ID:** `dan_kennedy_god_of_direct_response_marketing`
**Desired Slug:** `dan-kennedy`

**Step 1: Add slug mapping**
```python
# In generate_persona_skills.py
'dan_kennedy_god_of_direct_response_marketing': 'dan-kennedy',
```

**Step 2: Generate skill**
```bash
poetry run python -m dk_rag.scripts.generate_persona_skills \
  --persona-id dan_kennedy_god_of_direct_response_marketing \
  --output-dir /Users/blickt/.claude/skills
```

**Step 3: Verify**
```bash
ls /Users/blickt/.claude/skills/dan-kennedy/SKILL.md
# Should output: /Users/blickt/.claude/skills/dan-kennedy/SKILL.md
```

**Usage in Claude Code:** `/dan-kennedy`

---

### Example 2: Adding "Starter Story" Persona

**Persona ID:** `starter_story_youtuber`
**Desired Slug:** `starter-story`

**Step 1: Add slug mapping**
```python
# In generate_persona_skills.py
'starter_story_youtuber': 'starter-story',
```

**Step 2: Generate skill**
```bash
poetry run python -m dk_rag.scripts.generate_persona_skills \
  --persona-id starter_story_youtuber \
  --output-dir /Users/blickt/.claude/skills
```

**Output:**
```
11:21:36 - INFO - [ArtifactDisc] Found 1 artifacts for persona 'starter_story_youtuber'
11:21:36 - INFO - [ArtifactDisc] Latest artifact for 'starter_story_youtuber': persona_starter_story_youtuber_20251108_065927.json.gz
✓ Generated Skill for: starter_story_youtuber
```

**Usage in Claude Code:** `/starter-story`

---

### Example 3: Generating All Skills at Once

If you have multiple new personas, you can generate all skills in one command:

```bash
poetry run python -m dk_rag.scripts.generate_persona_skills \
  --output-dir /Users/blickt/.claude/skills
```

This will:
- Read all personas from persona_registry.json
- Generate a skill for each one
- Skip any that fail with an error message
- Report success/failure for each

---

## Advanced Usage

### Regenerating an Existing Skill

To update a skill after modifying the persona artifact:

```bash
# Simply re-run the generator - it will overwrite
poetry run python -m dk_rag.scripts.generate_persona_skills \
  --persona-id <persona_id> \
  --output-dir /Users/blickt/.claude/skills
```

### Custom Output Directory

To generate skills in a different location:

```bash
poetry run python -m dk_rag.scripts.generate_persona_skills \
  --persona-id <persona_id> \
  --output-dir /custom/path/to/skills
```

### Batch Processing

Create a script to generate multiple personas:

```bash
#!/bin/bash

PERSONAS=(
  "starter_story_youtuber"
  "alex_hormozi_youtuber"
  "dan_kennedy_god_of_direct_response_marketing"
)

for persona in "${PERSONAS[@]}"; do
  echo "Generating skill for: $persona"
  poetry run python -m dk_rag.scripts.generate_persona_skills \
    --persona-id "$persona" \
    --output-dir /Users/blickt/.claude/skills
done
```

---

## Best Practices

1. **Slug Naming:**
   - Keep slugs short and memorable
   - Use the persona's brand name if well-known
   - Avoid abbreviations unless universally recognized

2. **Before Generating:**
   - Always verify persona artifact exists and is recent
   - Check persona_registry.json entry is complete
   - Ensure linguistic_style data is populated in artifact

3. **After Generating:**
   - Always verify the skill file visually
   - Test activation in Claude Code
   - Check that catchphrases and vocabulary are appropriate

4. **Version Control:**
   - Commit slug_map changes to git
   - Document why specific slugs were chosen
   - Don't commit the generated skills (they're in ~/.claude/)

5. **Maintenance:**
   - Regenerate skills when persona artifacts are updated
   - Keep slug_map organized alphabetically
   - Document custom slugs in team knowledge base

---

## Quick Reference

### One-Line Command Template
```bash
poetry run python -m dk_rag.scripts.generate_persona_skills --persona-id <ID> --output-dir /Users/blickt/.claude/skills
```

### File Paths Quick Copy
```
Script: /Users/blickt/Documents/src/pdf_2_text/dk_rag/scripts/generate_persona_skills.py
Output: /Users/blickt/.claude/skills/<slug>/SKILL.md
Registry: /Volumes/J15/aicallgo_data/persona_data_base/personas/persona_registry.json
Artifacts: /Volumes/J15/aicallgo_data/persona_data_base/personas/<persona_id>/artifacts/
```

### Essential Verification Commands
```bash
# Verify artifact exists
ls /Volumes/J15/aicallgo_data/persona_data_base/personas/<persona_id>/artifacts/

# Check skill was created
ls -lh /Users/blickt/.claude/skills/<slug>/SKILL.md

# View first 20 lines
head -20 /Users/blickt/.claude/skills/<slug>/SKILL.md
```

---

## Questions or Issues?

If you encounter problems not covered in this guide:

1. Check the error message for specific Python/module errors
2. Verify all prerequisites are met
3. Review the troubleshooting section
4. Check git history for recent changes to the generator script
5. Consult the dk_rag documentation for persona pipeline issues

---

**Last Updated:** November 8, 2025
**Script Version:** generate_persona_skills.py (as of starter-story persona addition)
