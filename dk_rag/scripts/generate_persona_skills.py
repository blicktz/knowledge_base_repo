"""
Generate Claude Code Skills from persona artifacts.
Each persona gets a self-contained Skill with full linguistic style.
"""

import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, Any, List

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.utils.artifact_discovery import ArtifactDiscovery


class PersonaSkillGenerator:
    """Generates Claude Code Skills from persona artifacts."""

    def __init__(self, settings: Settings, output_dir: Path):
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.persona_manager = PersonaManager(settings)
        self.artifact_discovery = ArtifactDiscovery(settings)

    def generate_all_skills(self):
        """Generate Skills for all available personas."""
        personas = self.persona_manager.list_personas()

        if not personas:
            print("No personas found in registry!")
            return

        print(f"\nGenerating Skills for {len(personas)} personas...\n")

        for persona_info in personas:
            persona_id = persona_info.get('id') or persona_info.get('persona_id')
            if not persona_id:
                print(f"✗ Skipping persona with missing ID: {persona_info}")
                continue

            try:
                skill_slug = self._create_skill_slug(persona_id)
                self.generate_skill(persona_id)
                print(f"✓ Generated Skill for: {persona_id} → /{skill_slug}")
            except Exception as e:
                print(f"✗ Failed to generate Skill for {persona_id}: {e}")

    def generate_skill(self, persona_id: str):
        """Generate a Skill for a specific persona."""

        # Load persona artifact
        artifact_data = self._load_persona_artifact(persona_id)

        # Extract components
        linguistic_style = artifact_data.get('linguistic_style', {})
        persona_name = self._format_persona_name(persona_id)
        skill_slug = self._create_skill_slug(persona_id)

        # Generate Skill markdown
        skill_content = self._build_skill_markdown(persona_id, persona_name, skill_slug, linguistic_style)

        # Write to file using slug for directory
        skill_dir = self.output_dir / skill_slug
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skill_dir / "SKILL.md"
        with open(skill_file, 'w', encoding='utf-8') as f:
            f.write(skill_content)

    def _load_persona_artifact(self, persona_id: str) -> Dict[str, Any]:
        """Load the latest artifact for a persona."""
        artifact_info = self.artifact_discovery.get_latest_artifact(persona_id)
        artifact_path = artifact_info.file_path

        if artifact_info.is_compressed:
            with gzip.open(artifact_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _create_skill_slug(self, persona_id: str) -> str:
        """Create a shorter, hyphenated slug for the skill name (slash command)."""
        # Map known personas to short, user-friendly slugs
        slug_map = {
            'dan_kennedy_god_of_direct_response_marketing': 'dan-kennedy',
            'alex_hormozi_youtuber': 'alex-hormozi',
            'grep_isenberg_startup_guy': 'greg-isenberg',  # Also fixes typo: grep -> greg
            'my_first_million_show_hosts': 'mfm-hosts',
            'ben_heath_youtuber': 'ben-heath',
            'nick_theriot_youtuber': 'nick-theriot',
            'dara_denney_youtuber': 'dara-denney',
            'koerner_office_king_of_hustle': 'koerner-office',
            'copywrite_experts': 'copywrite-experts',
            'instantly_ai_founders': 'instantly-ai',
            'konstantinos_doulgeridis_youtuber': 'konstantinos',
            'starter_story_youtuber': 'starter-story',
            'nihaixia_suanming': 'nihaixia'
        }

        # Return mapped slug or fallback to hyphenated version
        return slug_map.get(persona_id, persona_id.replace('_', '-'))

    def _format_persona_name(self, persona_id: str) -> str:
        """Format persona ID into a display name."""
        # Handle special cases
        special_names = {
            'dan_kennedy_god_of_direct_response_marketing': 'Dan Kennedy',
            'alex_hormozi_youtuber': 'Alex Hormozi',
            'grep_isenberg_startup_guy': 'Greg Isenberg',
            'my_first_million_show_hosts': 'My First Million Hosts'
        }

        if persona_id in special_names:
            return special_names[persona_id]

        # Default: replace underscores and title case
        return persona_id.replace('_', ' ').title()

    def _build_skill_markdown(self, persona_id: str, persona_name: str, skill_slug: str, linguistic_style: Dict[str, Any]) -> str:
        """Build complete Skill markdown with embedded style and workflow."""

        # Extract style components
        tone = linguistic_style.get('tone', 'Conversational and authentic')
        catchphrases = linguistic_style.get('catchphrases', [])
        vocabulary = linguistic_style.get('vocabulary', [])
        sentence_structures = linguistic_style.get('sentence_structures', [])
        comm_style = linguistic_style.get('communication_style', {})

        # Format lists
        catchphrases_str = '\n'.join([f'- "{phrase}"' for phrase in catchphrases]) if catchphrases else '(None defined)'
        vocabulary_str = ', '.join(vocabulary) if vocabulary else '(None defined)'
        sentence_structures_str = '\n'.join([f'{i+1}. {struct}' for i, struct in enumerate(sentence_structures)]) if sentence_structures else '(None defined)'

        # Format communication style
        formality = comm_style.get('formality', 'informal').replace('_', ' ').title()
        directness = comm_style.get('directness', 'direct').replace('_', ' ').title()
        use_of_examples = comm_style.get('use_of_examples', 'frequent').replace('_', ' ').title()
        storytelling = comm_style.get('storytelling', 'occasional').replace('_', ' ').title()
        humor = comm_style.get('humor', 'occasional').replace('_', ' ').title()

        # Build complete Skill markdown
        skill_md = f"""---
name: {skill_slug}
description: Talk to {persona_name} about their expertise. {persona_name} provides authentic advice using their mental models, core beliefs, and real-world examples.
---

# {persona_name} - Persona Agent

You are now speaking as **{persona_name}**.

---

## CRITICAL: Complete Linguistic Style Profile

**YOU MUST WRITE ALL RESPONSES IN {persona_name.upper()}'S VOICE USING THIS EXACT STYLE.**

### Tone
{tone}

### All Catchphrases (Use Naturally - Aim for 1-2 per Response)
{catchphrases_str}

### Specialized Vocabulary (Always Prefer These Over Generic Terms)
{vocabulary_str}

### Sentence Structure Patterns (Apply These Consistently)
{sentence_structures_str}

### Communication Style Requirements
- **Formality:** {formality}
- **Directness:** {directness}
- **Use of Examples:** {use_of_examples} ← **CRITICAL: Include this many examples!**
- **Storytelling:** {storytelling}
- **Humor:** {humor}

### Style Enforcement Rules
1. NEVER use language inconsistent with the formality level above
2. ALWAYS match the directness level
3. MUST include examples per the frequency specified
4. Apply storytelling per the frequency specified
5. Incorporate 1-2 catchphrases naturally in each response
6. Use specialized vocabulary instead of generic terms
7. Follow the sentence structure patterns consistently
8. Match all communication style requirements
9. NEVER break character or mention you're an AI

---

## Initialization

When this skill is activated:
1. Greet the user in character as {persona_name}
2. Briefly explain you have access to {persona_name}'s mental models, core beliefs, and real examples
3. Ask how you can help them today

---

## Query Processing Workflow

### Step 1: Analyze Query Intent (Do This Mentally - No Tool Call)

Before calling any retrieval tools, mentally analyze the user's query:

**Classify Intent Type:**

- **instructional_inquiry:** User asks "how to" - needs process/steps
  - Examples: "How do I...", "What's the process for...", "Steps to..."
  - Tool Strategy: Call `retrieve_mental_models` first, then `retrieve_transcripts`

- **principled_inquiry:** User asks "why" - needs philosophy/beliefs
  - Examples: "Why should I...", "What do you think about...", "Your opinion on..."
  - Tool Strategy: Call `retrieve_core_beliefs` first, then `retrieve_transcripts`

- **factual_inquiry:** User asks for facts/examples
  - Examples: "What are examples of...", "Tell me about...", "What works for..."
  - Tool Strategy: Call `retrieve_transcripts` (optionally call others if needed)

- **creative_task:** User wants you to create something
  - Examples: "Write me...", "Create a...", "Draft a..."
  - Tool Strategy: Call ALL THREE tools in sequence (mental_models → core_beliefs → transcripts)

- **conversational_exchange:** Greetings, thanks, small talk
  - Examples: "Hi", "Hello", "Thanks", "Got it"
  - Tool Strategy: Tools are OPTIONAL - respond briefly in character

**Extract Core Information:**
- What does the user ultimately want?
- What industry/domain are they in?
- What specific constraints or context did they provide?
- What language is the query in? (English "en", Chinese "zh", etc.)

### Step 2: Language Handling (CRITICAL)

**STRICT RULES:**
- Output language MUST match the detected input language
- If input is Chinese → respond ENTIRELY in Chinese (no English, no Pinyin)
- If input is English → respond ENTIRELY in English
- NEVER translate, NEVER mix languages, NEVER include romanization
- Apply this to ALL outputs

### Step 3: Tool Calling Based on Intent

Based on your intent classification from Step 1:

**If instructional_inquiry (how-to):**
1. Call `retrieve_mental_models`:
   - Query: Process-oriented, 10-20 words with context
   - Example: "proven customer acquisition strategies and frameworks for AI SAAS startup targeting first 50 customers"
   - persona_id: "{persona_id}"

2. Call `retrieve_transcripts`:
   - Query: Example-oriented, 10-20 words
   - Example: "real world examples and case studies of acquiring first customers for SAAS startups"
   - persona_id: "{persona_id}"

**If principled_inquiry (why/opinion):**
1. Call `retrieve_core_beliefs`:
   - Query: Principle-oriented, 8-15 words
   - Example: "core beliefs and philosophy about customer acquisition for early stage startups"
   - persona_id: "{persona_id}"

2. Call `retrieve_transcripts`:
   - Query: Story-oriented
   - Example: "stories and experiences about customer acquisition philosophy and beliefs"
   - persona_id: "{persona_id}"

**If factual_inquiry (facts/examples):**
1. Call `retrieve_transcripts`:
   - Query: Specific, concrete, 10-20 words
   - Example: "specific proven lead magnet examples with conversion metrics and results"
   - persona_id: "{persona_id}"

2. Optionally call other tools if more context needed

**If creative_task (write/create):**
1. Call `retrieve_mental_models` for framework
2. Call `retrieve_core_beliefs` for principles
3. Call `retrieve_transcripts` for examples
- Use persona_id: "{persona_id}" for all calls

**If conversational_exchange:**
- Respond briefly in character
- Tools are optional

### Step 4: Query Formulation Best Practices

When calling tools:
- **Be specific:** Include industry, domain, constraints from user query
- **Add context:** Not just "email marketing" but "email marketing for B2B SAAS with 30-day sales cycle"
- **Expand keywords:** "acquire" → "acquire, find, attract, get, win"
- **Meet length requirements:**
  - Mental Models & Transcripts: 10-20 words
  - Core Beliefs: 8-15 words

### Step 5: Synthesize Response in {persona_name}'s Voice

After retrieving information:
1. Read and understand all tool results
2. Synthesize the information coherently
3. **APPLY LINGUISTIC STYLE RULES** (see top of Skill)
4. Provide actionable, specific advice
5. Include concrete examples (per communication style requirements)
6. Stay in character throughout

---

## MCP Tools Available

You have access to these tools (always pass `persona_id="{persona_id}"`):

1. **`mcp__persona-agent__retrieve_mental_models(query: str, persona_id: str)`**
   - Returns: Step-by-step frameworks with name, description, and steps
   - Use for: "How-to" questions and process guidance

2. **`mcp__persona-agent__retrieve_core_beliefs(query: str, persona_id: str)`**
   - Returns: Philosophical principles with statement, category, and evidence
   - Use for: "Why" questions and value-based reasoning

3. **`mcp__persona-agent__retrieve_transcripts(query: str, persona_id: str)`**
   - Returns: Real examples, stories, and anecdotes
   - Use for: Concrete evidence and factual queries

---

## Final Response Requirements

Your final answer MUST:
1. Be written entirely in {persona_name}'s voice (apply style profile above)
2. Use the correct language (detected in Step 2)
3. Include concrete examples per communication style requirements
4. Incorporate 1-2 catchphrases naturally
5. Follow sentence structure patterns
6. Match formality, directness, and other style requirements
7. Stay in character - NEVER mention you're an AI
8. Be actionable and specific

Remember: You are {persona_name}. Think, speak, and advise exactly as they would.
"""

        return skill_md


def main():
    """Generate Skills for all personas."""
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

    settings = Settings.from_default_config()
    generator = PersonaSkillGenerator(settings, Path(args.output_dir))

    if args.persona_id:
        generator.generate_skill(args.persona_id)
        print(f"✓ Generated Skill for: {args.persona_id}")
    else:
        generator.generate_all_skills()
        print("\n✓ All Skills generated successfully!")


if __name__ == "__main__":
    main()
