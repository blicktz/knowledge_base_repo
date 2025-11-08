#!/usr/bin/env python3
"""
Streamlit Knowledge Browser for Virtual Influencer Persona Agent
Browse mental models and core beliefs with filtering and search
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dk_rag.config.settings import Settings
from dk_rag.data.storage.mental_models_store import MentalModelsStore
from dk_rag.data.storage.core_beliefs_store import CoreBeliefsStore
from dk_rag.data.models.persona_constitution import MentalModel, CoreBelief
from dk_rag.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeBrowser:
    """Streamlit-based browser for persona knowledge."""

    def __init__(self):
        self.settings = Settings.from_default_config()
        self.available_personas = self._discover_personas()
        logger.info(f"Knowledge browser initialized with personas: {self.available_personas}")

    def _discover_personas(self) -> List[str]:
        """Discover available personas from the personas directory."""
        personas_dir = Path(self.settings.get_personas_base_dir())
        if not personas_dir.exists():
            logger.warning(f"Personas directory not found: {personas_dir}")
            return []

        personas = []
        for persona_dir in personas_dir.iterdir():
            if persona_dir.is_dir() and persona_dir.name != "__pycache__":
                personas.append(persona_dir.name)

        return sorted(personas) if personas else []

    def _load_mental_models(self, persona_id: str) -> List[Dict[str, Any]]:
        """Load all mental models for a persona from vector store."""
        try:
            # Initialize mental models store
            store = MentalModelsStore(self.settings, persona_id)

            # Check if collection exists
            if not store.collection_exists():
                logger.warning(f"No mental models collection found for {persona_id}")
                return []

            # Get all documents from the collection
            # We'll do a broad search to get all items
            all_docs = store.collection.get()

            # Parse documents into mental models format
            mental_models = []
            if all_docs and 'metadatas' in all_docs:
                for i, metadata in enumerate(all_docs['metadatas']):
                    # Get the document text
                    doc_text = all_docs['documents'][i] if 'documents' in all_docs else ""

                    # Build mental model dict from metadata
                    # Note: ChromaDB stores lists as delimited strings
                    model = {
                        'name': metadata.get('name', f'Model {i+1}'),
                        'description': metadata.get('description', doc_text),
                        'steps': [s.strip() for s in metadata.get('steps_text', '').split('\n') if s.strip()],
                        'categories': [c.strip() for c in metadata.get('categories_text', '').split('|') if c.strip()],
                        'confidence_score': metadata.get('confidence_score', 0.0),
                        'frequency': metadata.get('frequency', 1)
                    }
                    mental_models.append(model)

            logger.info(f"Loaded {len(mental_models)} mental models for {persona_id}")
            return mental_models

        except Exception as e:
            logger.error(f"Failed to load mental models for {persona_id}: {e}")
            return []

    def _load_core_beliefs(self, persona_id: str) -> List[Dict[str, Any]]:
        """Load all core beliefs for a persona from vector store."""
        try:
            # Initialize core beliefs store
            store = CoreBeliefsStore(self.settings, persona_id)

            # Check if collection exists
            if not store.collection_exists():
                logger.warning(f"No core beliefs collection found for {persona_id}")
                return []

            # Get all documents from the collection
            all_docs = store.collection.get()

            # Parse documents into core beliefs format
            core_beliefs = []
            if all_docs and 'metadatas' in all_docs:
                for i, metadata in enumerate(all_docs['metadatas']):
                    # Get the document text
                    doc_text = all_docs['documents'][i] if 'documents' in all_docs else ""

                    # Build core belief dict from metadata
                    # Note: ChromaDB stores lists as delimited strings
                    belief = {
                        'statement': metadata.get('statement', doc_text),
                        'category': metadata.get('category', 'general'),
                        'confidence_score': metadata.get('confidence_score', 0.0),
                        'frequency': metadata.get('frequency', 1),
                        'supporting_evidence': [e.strip() for e in metadata.get('supporting_evidence_text', '').split('\n') if e.strip()]
                    }
                    core_beliefs.append(belief)

            logger.info(f"Loaded {len(core_beliefs)} core beliefs for {persona_id}")
            return core_beliefs

        except Exception as e:
            logger.error(f"Failed to load core beliefs for {persona_id}: {e}")
            return []

    def _get_all_categories(self, items: List[Dict[str, Any]], knowledge_type: str) -> List[str]:
        """Extract all unique categories from items."""
        categories = set()
        for item in items:
            if knowledge_type == "Mental Models":
                item_categories = item.get('categories', [])
                if isinstance(item_categories, list):
                    categories.update(item_categories)
            else:  # Core Beliefs
                category = item.get('category', 'general')
                if category:
                    categories.add(category)
        return sorted(list(categories))

    def _filter_items(
        self,
        items: List[Dict[str, Any]],
        search_query: str,
        selected_categories: List[str],
        min_confidence: float,
        knowledge_type: str
    ) -> List[Dict[str, Any]]:
        """Filter items based on search query, categories, and confidence score."""
        filtered = []

        for item in items:
            # Filter by confidence score
            confidence = item.get('confidence_score', 0.0)
            if confidence < min_confidence:
                continue

            # Filter by categories
            if selected_categories:
                if knowledge_type == "Mental Models":
                    item_categories = item.get('categories', [])
                    if not any(cat in selected_categories for cat in item_categories):
                        continue
                else:  # Core Beliefs
                    item_category = item.get('category', 'general')
                    if item_category not in selected_categories:
                        continue

            # Filter by search query
            if search_query:
                query_lower = search_query.lower()
                if knowledge_type == "Mental Models":
                    name = item.get('name', '').lower()
                    description = item.get('description', '').lower()
                    if query_lower not in name and query_lower not in description:
                        continue
                else:  # Core Beliefs
                    statement = item.get('statement', '').lower()
                    if query_lower not in statement:
                        continue

            filtered.append(item)

        return filtered

    def _render_mental_model(self, model: Dict[str, Any]):
        """Render a single mental model as an article-like card."""
        name = model.get('name', 'Unnamed Model')
        description = model.get('description', 'No description available')
        steps = model.get('steps', [])
        categories = model.get('categories', [])
        confidence = model.get('confidence_score', 0.0)
        frequency = model.get('frequency', 0)

        # Create an expander for each mental model
        with st.expander(f"**{name}**", expanded=False):
            # Categories as tags if available
            if categories:
                category_tags = " • ".join([f"`{cat}`" for cat in categories])
                st.markdown(category_tags)
                st.markdown("")  # Add spacing

            # Description
            st.markdown(description)

            # Steps
            if steps:
                st.markdown("")  # Add spacing
                st.markdown("**Steps:**")
                for i, step in enumerate(steps, 1):
                    st.markdown(f"{i}. {step}")

            # Footer with metadata
            st.markdown("")  # Add spacing
            st.caption(f"Confidence: {confidence:.1%} • Frequency: {frequency}")

    def _render_core_belief(self, belief: Dict[str, Any]):
        """Render a single core belief as an article-like card."""
        statement = belief.get('statement', 'No statement available')
        category = belief.get('category', 'general')
        confidence = belief.get('confidence_score', 0.0)
        frequency = belief.get('frequency', 0)
        evidence = belief.get('supporting_evidence', [])

        # Create an expander for each core belief
        with st.expander(f"**{statement[:100]}{'...' if len(statement) > 100 else ''}**", expanded=False):
            # Category as tag
            st.markdown(f"`{category}`")
            st.markdown("")  # Add spacing

            # Full statement
            st.markdown(statement)

            # Supporting evidence
            if evidence:
                st.markdown("")  # Add spacing
                st.markdown("**Supporting Evidence:**")
                for item in evidence:
                    st.markdown(f"- {item}")

            # Footer with metadata
            st.markdown("")  # Add spacing
            st.caption(f"Confidence: {confidence:.1%} • Frequency: {frequency}")

    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(
            page_title="Knowledge Browser",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Main title
        st.title("📚 Knowledge Browser")
        st.markdown("Browse mental models and core beliefs from persona knowledge bases")

        # Sidebar filters
        with st.sidebar:
            st.header("Filters")

            # Persona selector
            if not self.available_personas:
                st.error("No personas found. Please ensure personas are configured.")
                return

            selected_persona = st.selectbox(
                "Select Persona",
                options=self.available_personas,
                index=0
            )

            # Knowledge type selector
            knowledge_type = st.radio(
                "Knowledge Type",
                options=["Mental Models", "Core Beliefs"],
                index=0
            )

            st.divider()

            # Load data
            if knowledge_type == "Mental Models":
                items = self._load_mental_models(selected_persona)
            else:
                items = self._load_core_beliefs(selected_persona)

            # Show total count
            st.info(f"Total: {len(items)} items")

            # Search box
            search_query = st.text_input(
                "Search",
                placeholder=f"Search {knowledge_type.lower()}...",
                help="Filter by name/description for Mental Models, or statement for Core Beliefs"
            )

            # Category filter
            all_categories = self._get_all_categories(items, knowledge_type)
            if all_categories:
                selected_categories = st.multiselect(
                    "Filter by Category",
                    options=all_categories,
                    default=[]
                )
            else:
                selected_categories = []

            # Confidence score filter
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Filter items by minimum confidence score"
            )

            # Sort options
            sort_by = st.selectbox(
                "Sort By",
                options=["Confidence (High to Low)", "Confidence (Low to High)", "Frequency (High to Low)", "Frequency (Low to High)"],
                index=0
            )

        # Main content area
        if not items:
            st.warning(f"No {knowledge_type.lower()} found for {selected_persona}")
            return

        # Filter items
        filtered_items = self._filter_items(
            items,
            search_query,
            selected_categories,
            min_confidence,
            knowledge_type
        )

        # Sort items
        if sort_by == "Confidence (High to Low)":
            filtered_items = sorted(filtered_items, key=lambda x: x.get('confidence_score', 0.0), reverse=True)
        elif sort_by == "Confidence (Low to High)":
            filtered_items = sorted(filtered_items, key=lambda x: x.get('confidence_score', 0.0))
        elif sort_by == "Frequency (High to Low)":
            filtered_items = sorted(filtered_items, key=lambda x: x.get('frequency', 0), reverse=True)
        elif sort_by == "Frequency (Low to High)":
            filtered_items = sorted(filtered_items, key=lambda x: x.get('frequency', 0))

        # Display results count
        st.markdown(f"### Showing {len(filtered_items)} of {len(items)} {knowledge_type}")

        if not filtered_items:
            st.info("No items match your filters. Try adjusting your search criteria.")
            return

        # Render items
        if knowledge_type == "Mental Models":
            for model in filtered_items:
                self._render_mental_model(model)
        else:
            for belief in filtered_items:
                self._render_core_belief(belief)


def main():
    """Entry point for the knowledge browser."""
    try:
        browser = KnowledgeBrowser()
        browser.run()
    except Exception as e:
        logger.error(f"Failed to start knowledge browser: {e}", exc_info=True)
        st.error(f"Failed to start knowledge browser: {e}")


if __name__ == "__main__":
    main()
