#!/usr/bin/env python3
"""
Verification Script: Check for Duplicate Mental Models and Core Beliefs

This script checks the ChromaDB collections for duplicate entries by:
1. Querying all documents in the collection
2. Grouping by mental model/core belief names
3. Reporting any duplicates found
4. Checking metadata consistency (especially steps_text)

Usage:
    python scripts/verify_knowledge_duplicates.py --persona <persona_id>
    python scripts/verify_knowledge_duplicates.py --persona alex_hormozi_youtuber --knowledge mental_models
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dk_rag.config.settings import Settings
from dk_rag.data.storage.mental_models_store import MentalModelsStore
from dk_rag.data.storage.core_beliefs_store import CoreBeliefsStore


def check_duplicates(collection, knowledge_type: str) -> Dict[str, Any]:
    """
    Check for duplicate entries in a ChromaDB collection.

    Args:
        collection: ChromaDB collection to check
        knowledge_type: Type of knowledge ('mental_models' or 'core_beliefs')

    Returns:
        Dictionary with duplicate analysis
    """
    print(f"\n{'='*80}")
    print(f"Checking {knowledge_type} for duplicates...")
    print(f"{'='*80}\n")

    # Get all documents
    try:
        # Query with very high k to get all documents
        all_docs = collection.similarity_search("", k=10000)
        total_docs = len(all_docs)
        print(f"✓ Retrieved {total_docs} total documents from ChromaDB\n")
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return {"error": str(e)}

    # Group by name
    documents_by_name = defaultdict(list)
    for i, doc in enumerate(all_docs):
        name = doc.metadata.get('name', f'unnamed_{i}')
        documents_by_name[name].append({
            'index': i,
            'doc': doc,
            'metadata': doc.metadata
        })

    # Analyze duplicates
    duplicates = {}
    unique_names = len(documents_by_name)

    for name, docs in documents_by_name.items():
        if len(docs) > 1:
            duplicates[name] = docs

    # Report results
    print(f"📊 Analysis Results:")
    print(f"   Total documents: {total_docs}")
    print(f"   Unique names: {unique_names}")
    print(f"   Duplicates found: {len(duplicates)}\n")

    if duplicates:
        print(f"⚠️  DUPLICATES DETECTED:\n")
        for name, docs in duplicates.items():
            print(f"   '{name}' appears {len(docs)} times:")

            for i, doc_info in enumerate(docs):
                doc = doc_info['doc']
                metadata = doc_info['metadata']

                # Check steps_text for mental models
                if knowledge_type == 'mental_models':
                    steps_text = metadata.get('steps_text', '')
                    steps_count = len([s for s in steps_text.split('\n') if s.strip()]) if steps_text else 0
                    print(f"      [{i+1}] Index {doc_info['index']}: steps_text={len(steps_text)} chars, {steps_count} steps")

                # Check supporting_evidence_text for core beliefs
                elif knowledge_type == 'core_beliefs':
                    evidence_text = metadata.get('supporting_evidence_text', '')
                    evidence_count = len([e for e in evidence_text.split('\n') if e.strip()]) if evidence_text else 0
                    print(f"      [{i+1}] Index {doc_info['index']}: evidence_text={len(evidence_text)} chars, {evidence_count} items")

            print()
    else:
        print(f"✅ No duplicates found! All {unique_names} {knowledge_type} have unique entries.\n")

    # Check for missing metadata
    print(f"🔍 Metadata Integrity Check:")
    missing_steps = []
    missing_evidence = []

    for name, docs_list in documents_by_name.items():
        # Check first instance (should be the only one)
        doc_info = docs_list[0]
        metadata = doc_info['metadata']

        if knowledge_type == 'mental_models':
            steps_text = metadata.get('steps_text', '')
            if not steps_text:
                missing_steps.append(name)

        elif knowledge_type == 'core_beliefs':
            evidence_text = metadata.get('supporting_evidence_text', '')
            if not evidence_text:
                missing_evidence.append(name)

    if knowledge_type == 'mental_models':
        if missing_steps:
            print(f"   ⚠️  {len(missing_steps)} {knowledge_type} have empty steps_text:")
            for name in missing_steps[:10]:
                print(f"      - {name}")
            if len(missing_steps) > 10:
                print(f"      ... and {len(missing_steps) - 10} more")
        else:
            print(f"   ✅ All {unique_names} mental models have steps_text populated")

    elif knowledge_type == 'core_beliefs':
        if missing_evidence:
            print(f"   ⚠️  {len(missing_evidence)} {knowledge_type} have empty supporting_evidence_text:")
            for name in missing_evidence[:10]:
                print(f"      - {name}")
            if len(missing_evidence) > 10:
                print(f"      ... and {len(missing_evidence) - 10} more")
        else:
            print(f"   ✅ All {unique_names} core beliefs have supporting_evidence_text populated")

    print()

    return {
        'total_documents': total_docs,
        'unique_names': unique_names,
        'duplicate_count': len(duplicates),
        'duplicates': list(duplicates.keys()),
        'missing_metadata': missing_steps if knowledge_type == 'mental_models' else missing_evidence
    }


def main():
    parser = argparse.ArgumentParser(description='Verify knowledge base for duplicates')
    parser.add_argument('--persona', required=True, help='Persona ID')
    parser.add_argument(
        '--knowledge',
        choices=['mental_models', 'core_beliefs', 'both'],
        default='both',
        help='Knowledge type to check'
    )

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# Knowledge Base Duplicate Verification")
    print(f"# Persona: {args.persona}")
    print(f"{'#'*80}")

    # Initialize settings
    settings = Settings()

    results = {}

    # Check mental models
    if args.knowledge in ['mental_models', 'both']:
        try:
            mm_store = MentalModelsStore(
                settings=settings,
                persona_id=args.persona
            )

            results['mental_models'] = check_duplicates(
                mm_store.collection,
                'mental_models'
            )
        except Exception as e:
            print(f"❌ Error checking mental models: {e}\n")
            results['mental_models'] = {'error': str(e)}

    # Check core beliefs
    if args.knowledge in ['core_beliefs', 'both']:
        try:
            cb_store = CoreBeliefsStore(
                settings=settings,
                persona_id=args.persona
            )

            results['core_beliefs'] = check_duplicates(
                cb_store.collection,
                'core_beliefs'
            )
        except Exception as e:
            print(f"❌ Error checking core beliefs: {e}\n")
            results['core_beliefs'] = {'error': str(e)}

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    for knowledge_type, result in results.items():
        if 'error' in result:
            print(f"❌ {knowledge_type}: Error - {result['error']}")
        else:
            status = "✅ PASS" if result['duplicate_count'] == 0 else "⚠️  FAIL"
            print(f"{status} {knowledge_type}:")
            print(f"   Total: {result['total_documents']}")
            print(f"   Unique: {result['unique_names']}")
            print(f"   Duplicates: {result['duplicate_count']}")
            print(f"   Missing metadata: {len(result.get('missing_metadata', []))}")

    print()

    # Exit with error code if duplicates found
    has_duplicates = any(
        r.get('duplicate_count', 0) > 0
        for r in results.values()
        if 'error' not in r
    )

    if has_duplicates:
        print("⚠️  Duplicates detected! Please rebuild the knowledge base.\n")
        sys.exit(1)
    else:
        print("✅ All checks passed! No duplicates found.\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
