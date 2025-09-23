"""
Utility functions for working with LLM responses
"""

import re
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import Any, Optional, Tuple, Dict
import logging

try:
    from lxml import etree as lxml_etree, html as lxml_html
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False


def clean_llm_json_response(text: str) -> str:
    """
    Clean LLM response text to extract pure JSON content.
    
    Simple and robust approach using direct markdown stripping.
    Handles common LLM response formats:
    - XML wrapped: <json_output>content</json_output>
    - Markdown blocks: ```json\n{content}\n``` or ```\n{content}\n```
    - Plain JSON (passthrough)
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned JSON string ready for json.loads()
        
    Raises:
        ValueError: If no valid JSON content can be extracted
    """
    if not text or not text.strip():
        raise ValueError("Empty or whitespace-only text provided")
    
    text = text.strip()
    
    # Handle XML wrapping first: <json_output>content</json_output>
    xml_match = re.search(r'<json_output>\s*(.*?)\s*</json_output>', text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        text = xml_match.group(1).strip()
    
    # Use the proven _strip_markdown_blocks function to handle all markdown variants
    cleaned = _strip_markdown_blocks(text)
    return cleaned if cleaned != text else text


def _strip_markdown_blocks(text: str) -> str:
    """
    Strip markdown code blocks from text.
    
    Handles:
    - ```json\n{content}\n```
    - ```\n{content}\n```
    - ````json\n{content}\n````
    """
    if not text:
        return text
    
    # Pattern for markdown code blocks with optional language specification
    # Supports both ``` and ```` variants
    patterns = [
        r'^```+(?:json)?\s*\n?(.*?)\n?\s*```+$',  # Full blocks
        r'^```+(?:json)?\s*(.*?)\s*```+$',        # Without newlines
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Validate this looks like JSON (starts with { or [)
            if content and content[0] in '{[':
                return content
    
    return text


def safe_json_loads(text: str) -> Any:
    """
    Safely parse JSON from LLM response with automatic cleanup.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Parsed JSON object
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after cleanup
        ValueError: If text is empty or invalid
    """
    try:
        # First try direct parsing (fastest path)
        return json.loads(text)
    except json.JSONDecodeError:
        # Try with cleanup
        cleaned_text = clean_llm_json_response(text)
        return json.loads(cleaned_text)


def extract_json_from_xml_response(response_text: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Extract JSON content from LLM XML response using robust parsing.
    
    Uses xml.etree.ElementTree for proper XML parsing with comprehensive fallback strategies.
    Handles common LLM XML tag variations and malformed XML gracefully.
    
    Args:
        response_text: Raw LLM response text that may contain XML
        logger: Optional logger for detailed extraction reporting
        
    Returns:
        Tuple of (extracted_json_text, extraction_metadata)
        - extracted_json_text: Cleaned JSON string or None if extraction failed
        - extraction_metadata: Dict with extraction details for debugging
    """
    if not response_text or not response_text.strip():
        return None, {"method": "none", "error": "Empty response text"}
    
    metadata = {
        "original_length": len(response_text),
        "method": "unknown",
        "xml_tags_found": [],
        "warnings": [],
        "success": False
    }
    
    # Stage 1: Try proper XML parsing with ElementTree
    json_content = _try_xml_parsing(response_text, metadata, logger)
    if json_content:
        metadata["method"] = "xml_parsing"
        metadata["success"] = True
        return json_content, metadata
    
    # Stage 2: Fall back to existing clean_llm_json_response function
    try:
        json_content = clean_llm_json_response(response_text)
        if json_content and json_content != response_text:
            metadata["method"] = "markdown_cleanup" 
            metadata["success"] = True
            if logger:
                logger.info("XML parsing failed, markdown cleanup succeeded")
            return json_content, metadata
    except Exception as e:
        metadata["warnings"].append(f"Markdown cleanup failed: {str(e)}")
    
    # Stage 3: Try regex extraction of largest JSON block
    json_content = _try_regex_json_extraction(response_text, metadata, logger)
    if json_content:
        metadata["method"] = "regex_extraction"
        metadata["success"] = True
        return json_content, metadata
    
    # Stage 4: Complete failure
    metadata["method"] = "failed"
    metadata["error"] = "All extraction methods failed"
    if logger:
        logger.error(f"All JSON extraction methods failed. Response sample: {response_text[:500]}...")
    
    return None, metadata


def _try_xml_parsing(response_text: str, metadata: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Attempt to parse response as XML and extract JSON content.
    Handles both nested structures (output_block > json_output) and direct structures.
    """
    # Strategy 1: Try lxml with recovery mode (most robust)
    if LXML_AVAILABLE:
        json_content = _try_lxml_parsing(response_text, metadata, logger)
        if json_content:
            return json_content
    
    # Strategy 2: Handle nested structure (output_block containing json_output)
    json_content = _try_nested_xml_parsing(response_text, metadata, logger)
    if json_content:
        return json_content
    
    # Strategy 3: Try direct XML tag extraction (legacy approach)
    return _try_direct_xml_parsing(response_text, metadata, logger)


def _try_lxml_parsing(response_text: str, metadata: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Attempt to parse response using lxml with recovery mode for maximum robustness.
    Handles malformed XML that would break ElementTree parsing.
    """
    if not LXML_AVAILABLE:
        metadata["warnings"].append("lxml not available for robust parsing")
        return None
    
    # Common XML tag variations used by LLMs
    json_tag_patterns = [
        'json_output',
        'output_block', 
        'json',
        'response',
        'result',
        'output'
    ]
    
    # Strategy 1: Try XML parsing with recovery mode
    try:
        # Create a recovery parser that attempts to fix malformed XML
        parser = lxml_etree.XMLParser(recover=True, strip_cdata=False)
        
        # Check for common LLM response patterns that need wrapping
        needs_wrapping = (
            ('<thinking>' in response_text and '<json_output>' in response_text) or
            ('<thinking>' in response_text and '</thinking>' in response_text and response_text.strip().find('</thinking>') < len(response_text.strip()) - 12)
        )
        
        if needs_wrapping:
            # Wrap in a root element for proper XML structure
            wrapped_xml = f"<root>{response_text}</root>"
            root = lxml_etree.fromstring(wrapped_xml.encode('utf-8'), parser)
            if logger:
                logger.debug("lxml: Successfully parsed XML with root wrapper (LLM response pattern detected)")
        else:
            # Try parsing as-is first
            try:
                root = lxml_etree.fromstring(response_text.encode('utf-8'), parser)
                if logger:
                    logger.debug("lxml: Successfully parsed XML as-is")
            except Exception:
                # If that fails, wrap in a root element
                wrapped_xml = f"<root>{response_text}</root>"
                root = lxml_etree.fromstring(wrapped_xml.encode('utf-8'), parser)
                if logger:
                    logger.debug("lxml: Successfully parsed XML with root wrapper (fallback)")
        
        # Try to find JSON content using xpath
        for tag_name in json_tag_patterns:
            # Look for the tag anywhere in the tree
            elements = root.xpath(f'.//{tag_name}')
            if elements:
                json_content = elements[0].text
                if json_content and json_content.strip():
                    cleaned_content = json_content.strip()
                    # Additional validation - check if it looks like JSON
                    if cleaned_content.startswith(('{', '[')):
                        if logger:
                            logger.debug(f"lxml: Found JSON content in <{tag_name}> tag")
                        metadata["xml_tag_used"] = f"{tag_name} (lxml)"
                        metadata["xml_structure"] = "lxml_recovery"
                        return cleaned_content
        
        # If no content found in expected tags, check if root has text content
        if root.text and root.text.strip():
            text_content = root.text.strip()
            # Quick validation - check if it looks like JSON
            if text_content.startswith(('{', '[')):
                if logger:
                    logger.debug("lxml: Found JSON content in root text")
                metadata["xml_tag_used"] = "root_text (lxml)"
                metadata["xml_structure"] = "lxml_recovery"
                return text_content
        
        metadata["warnings"].append("lxml parsing succeeded but found no JSON content")
        return None
        
    except Exception as e:
        metadata["warnings"].append(f"lxml XML parsing failed: {str(e)}")
        if logger:
            logger.debug(f"lxml XML parsing failed: {str(e)}")
    
    # Strategy 2: Try HTML parsing (even more forgiving)
    try:
        # lxml's HTML parser is very forgiving and can handle broken XML-like structures
        doc = lxml_html.fromstring(response_text)
        
        for tag_name in json_tag_patterns:
            elements = doc.xpath(f'.//{tag_name}')
            if elements:
                json_content = elements[0].text_content()
                if json_content and json_content.strip():
                    if logger:
                        logger.debug(f"lxml: Found JSON content in <{tag_name}> tag using HTML parser")
                    metadata["xml_tag_used"] = f"{tag_name} (lxml_html)"
                    metadata["xml_structure"] = "lxml_html_recovery"
                    return json_content.strip()
        
        metadata["warnings"].append("lxml HTML parsing succeeded but found no JSON content")
        return None
        
    except Exception as e:
        metadata["warnings"].append(f"lxml HTML parsing failed: {str(e)}")
        if logger:
            logger.debug(f"lxml HTML parsing failed: {str(e)}")
        return None


def _try_nested_xml_parsing(response_text: str, metadata: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Handle nested XML structure like <output_block><json_output>content</json_output></output_block>
    """
    # Check if we have the expected nested structure
    if not ("<output_block>" in response_text and "</output_block>" in response_text):
        metadata["warnings"].append("No output_block found for nested parsing")
        return None
    
    try:
        # Parse the response as-is (it should be valid XML)
        root = ET.fromstring(response_text)
        
        # Look for json_output within output_block
        json_output_element = root.find('json_output')
        if json_output_element is not None and json_output_element.text:
            json_content = json_output_element.text.strip()
            if json_content:
                if logger:
                    logger.debug("XML parsing succeeded using nested structure: output_block > json_output")
                metadata["xml_tag_used"] = "json_output (nested)"
                metadata["xml_structure"] = "nested"
                return json_content
        
        # Also check for other possible nested tags
        for tag_name in ['json', 'result', 'output']:
            element = root.find(tag_name) 
            if element is not None and element.text:
                json_content = element.text.strip()
                if json_content:
                    if logger:
                        logger.debug(f"XML parsing succeeded using nested structure: output_block > {tag_name}")
                    metadata["xml_tag_used"] = f"{tag_name} (nested)"
                    metadata["xml_structure"] = "nested"
                    return json_content
        
        metadata["warnings"].append("output_block found but no json content within")
        return None
        
    except ParseError as e:
        error_msg = f"Nested XML parsing failed: {str(e)}"
        metadata["warnings"].append(error_msg)
        if logger:
            logger.debug(f"Nested XML parsing failed: {str(e)}")
            logger.debug(f"Failed XML content (first 1000 chars): {response_text[:1000]}")
        return None
    except Exception as e:
        error_msg = f"Nested XML parsing error: {str(e)}"
        metadata["warnings"].append(error_msg)
        if logger:
            logger.debug(f"Nested XML parsing error: {str(e)}")
            logger.debug(f"Failed XML content (first 1000 chars): {response_text[:1000]}")
        return None


def _try_direct_xml_parsing(response_text: str, metadata: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Handle direct XML tags at root level (legacy approach with root wrapper)
    """
    # Common XML tag variations used by LLMs
    json_tag_patterns = [
        'json_output',
        'output_block', 
        'json',
        'response',
        'result',
        'output'
    ]
    
    # First, scan for which tags are present
    found_tags = []
    for tag in json_tag_patterns:
        if f"<{tag}>" in response_text and f"</{tag}>" in response_text:
            found_tags.append(tag)
    
    metadata["xml_tags_found"] = found_tags
    
    if not found_tags:
        metadata["warnings"].append("No XML tags found in response")
        return None
    
    # Try to parse as XML with root wrapper
    try:
        # Wrap response in root element for proper XML parsing
        wrapped_xml = f"<root>{response_text}</root>"
        if logger:
            logger.debug(f"Attempting to parse wrapped XML (first 1000 chars): {wrapped_xml[:1000]}")
        root = ET.fromstring(wrapped_xml)
        
        # Try each found tag in order of preference
        for tag_name in found_tags:
            element = root.find(tag_name)
            if element is not None and element.text:
                json_content = element.text.strip()
                if json_content:
                    if logger:
                        logger.debug(f"XML parsing succeeded using direct tag: <{tag_name}>")
                    metadata["xml_tag_used"] = f"{tag_name} (direct)"
                    metadata["xml_structure"] = "direct"
                    return json_content
        
        metadata["warnings"].append("XML tags found but contained no content")
        return None
        
    except ParseError as e:
        error_msg = f"Direct XML parsing failed: {str(e)}"
        metadata["warnings"].append(error_msg)
        if logger:
            logger.debug(f"Direct XML parsing failed: {str(e)}")
            logger.debug(f"Failed wrapped XML (first 1000 chars): {wrapped_xml[:1000] if 'wrapped_xml' in locals() else 'XML wrapping failed'}")
        return None
    except Exception as e:
        error_msg = f"Direct XML parsing error: {str(e)}"
        metadata["warnings"].append(error_msg)
        if logger:
            logger.debug(f"Direct XML parsing error: {str(e)}")
            logger.debug(f"Failed wrapped XML (first 1000 chars): {wrapped_xml[:1000] if 'wrapped_xml' in locals() else 'XML wrapping failed'}")
        return None


def _try_regex_json_extraction(response_text: str, metadata: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Try to extract the largest JSON block from response using regex patterns.
    """
    # Pattern to find JSON objects or arrays
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple object pattern
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Simple array pattern
        r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # Nested objects
        r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]'  # Nested arrays
    ]
    
    largest_json = ""
    largest_size = 0
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, response_text, re.DOTALL)
        for match in matches:
            candidate = match.group(0).strip()
            if len(candidate) > largest_size:
                # Quick validation - try to parse as JSON
                try:
                    json.loads(candidate)
                    largest_json = candidate
                    largest_size = len(candidate)
                except json.JSONDecodeError:
                    continue
    
    if largest_json:
        if logger:
            logger.info(f"Regex extraction found JSON block of {largest_size} characters")
        metadata["extracted_size"] = largest_size
        return largest_json
    
    metadata["warnings"].append("No valid JSON blocks found with regex")
    return None


def robust_json_loads(text: str, logger: Optional[logging.Logger] = None) -> Any:
    """
    Enhanced version of safe_json_loads with XML-aware extraction and detailed logging.
    
    Args:
        text: Raw LLM response text
        logger: Optional logger for extraction reporting
        
    Returns:
        Parsed JSON object
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after all extraction attempts
        ValueError: If text is empty or invalid
    """
    if not text or not text.strip():
        raise ValueError("Empty or whitespace-only text provided")
    
    # Try XML-aware extraction first
    extracted_json, extraction_metadata = extract_json_from_xml_response(text, logger)
    
    if logger and extraction_metadata.get("warnings"):
        for warning in extraction_metadata["warnings"]:
            logger.debug(f"JSON extraction warning: {warning}")
    
    if extracted_json:
        if logger:
            logger.debug(f"JSON extraction successful via {extraction_metadata['method']}")
        return json.loads(extracted_json)
    
    # Final attempt with original text
    if logger:
        logger.error("All JSON extraction methods failed, attempting direct parse")
    return json.loads(text)


def clean_reduce_phase_json_response(text: str) -> str:
    """
    Clean LLM response text specifically for reduce phase JSON extraction.
    
    Designed to handle reduce phase responses that may have content before/after
    the markdown code blocks, unlike the existing clean_llm_json_response which
    expects the code block to span the entire string.
    
    Args:
        text: Raw LLM response text from reduce phase
        
    Returns:
        Cleaned JSON string ready for json.loads()
        
    Raises:
        ValueError: If no valid JSON content can be extracted
    """
    if not text or not text.strip():
        raise ValueError("Empty or whitespace-only text provided")
    
    text = text.strip()
    
    # Handle XML wrapping first: <json_output>content</json_output>
    xml_match = re.search(r'<json_output>\s*(.*?)\s*</json_output>', text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        text = xml_match.group(1).strip()
    
    # Use specialized reduce phase markdown stripping
    cleaned = _strip_reduce_markdown_blocks(text)
    return cleaned if cleaned != text else text


def _strip_reduce_markdown_blocks(text: str) -> str:
    """
    Strip markdown code blocks from reduce phase responses.
    
    Unlike the existing _strip_markdown_blocks, this function looks for markdown
    blocks anywhere in the text, not just spanning the entire string. This handles
    reduce phase responses that may have additional content before/after the JSON.
    
    Handles:
    - ```json\n{content}\n``` (anywhere in text)
    - ```\n{content}\n``` (anywhere in text)
    - ````json\n{content}\n```` (anywhere in text)
    """
    if not text:
        return text
    
    # Patterns that find markdown blocks anywhere in the text (no ^ or $ anchors)
    # Prioritize json-tagged blocks first
    # Use non-greedy matching and allow for content after closing backticks
    patterns = [
        r'```json\s*\n?(.*?)\n?```',        # ```json ... ``` (no trailing space requirement)
        r'````json\s*\n?(.*?)\n?````',      # ````json ... ````
        r'```\s*\n?(.*?)\n?```',            # ``` ... ``` (generic, no trailing space requirement)
        r'````\s*\n?(.*?)\n?````',          # ```` ... ```` (generic)
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            
            # Validate this looks like JSON (starts with { or [)
            if content and content[0] in '{[':
                # Additional validation - try to parse it to ensure it's valid JSON structure
                try:
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    # If this match fails JSON parsing, try the next pattern
                    continue
    
    return text