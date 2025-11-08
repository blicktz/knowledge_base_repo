# Knowledge Browser

A Streamlit-based web interface for browsing mental models and core beliefs from persona knowledge bases.

## Features

- **Browse Mental Models & Core Beliefs**: Read through all mental models and core beliefs in an article-like format
- **Filter by Category**: Multi-select category filters for focused browsing
- **Search**: Text search to find specific models or beliefs by name/description
- **Sort Options**: Sort by confidence score or frequency
- **Confidence Threshold**: Slider to filter items by minimum confidence score
- **Responsive Layout**: Clean, readable interface optimized for content consumption

## Installation

Install Streamlit if not already installed:

```bash
pip install streamlit
```

Or add to your requirements.txt:
```
streamlit>=1.30.0
```

## Running the Browser

### Option 1: Direct Python

```bash
cd dk_rag/ui
streamlit run knowledge_browser.py
```

### Option 2: From Project Root

```bash
streamlit run dk_rag/ui/knowledge_browser.py
```

### Option 3: Custom Port

```bash
streamlit run dk_rag/ui/knowledge_browser.py --server.port 8505
```

## Usage

1. **Select Persona**: Choose a persona from the sidebar dropdown
2. **Choose Knowledge Type**: Toggle between Mental Models and Core Beliefs
3. **Filter**:
   - Use the search box to find specific content
   - Select categories to narrow down results
   - Adjust minimum confidence threshold
4. **Sort**: Choose your preferred sorting method
5. **Read**: Click on any item to expand and read the full content

## Data Source

The browser loads data directly from ChromaDB vector stores:
- Mental Models: `data/storage/personas/{persona_id}/vector_db_mental_models/`
- Core Beliefs: `data/storage/personas/{persona_id}/vector_db_core_beliefs/`

## UI Components

### Mental Models Display
- **Name**: Title of the mental model
- **Description**: Detailed explanation
- **Steps**: Step-by-step breakdown
- **Categories**: Applicable domains
- **Confidence Score**: Extraction confidence (0-100%)
- **Frequency**: Number of references found

### Core Beliefs Display
- **Statement**: The core belief statement
- **Category**: Belief category
- **Supporting Evidence**: Examples and evidence
- **Confidence Score**: Extraction confidence (0-100%)
- **Frequency**: Number of times expressed

## Tips

- Use the search box as a real-time filter - results update as you type
- Combine multiple filters for precise browsing
- Higher confidence scores generally indicate more reliable extractions
- Frequency indicates how often the model/belief appeared in source content

## Troubleshooting

### No personas found
- Ensure personas are configured in `data/storage/personas/`
- Check that vector stores have been initialized

### Empty results
- Verify that the persona has indexed mental models or core beliefs
- Check the minimum confidence slider - try setting it to 0.0
- Clear any active category filters

### Connection errors
- Ensure all dependencies are installed
- Check that ChromaDB vector stores are accessible
