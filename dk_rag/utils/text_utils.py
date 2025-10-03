"""
Text processing utilities for language-aware operations.

This module provides language-aware text processing functions for handling
both English and Chinese content accurately.
"""

from typing import List


def count_words(text: str, language: str = "en") -> int:
    """
    Count words in text with language-aware logic.

    For Chinese: Each character counts as one word (CJK characters don't use spaces)
    For English: Space-separated word counting

    Args:
        text: Text to count words in
        language: Language code ('en' for English, 'zh' for Chinese, etc.)

    Returns:
        Word count as integer

    Examples:
        >>> count_words("Hello world", "en")
        2
        >>> count_words("你好世界", "zh")
        4
    """
    if not text:
        return 0

    # Strip language parameter
    language = language.strip() if language else "en"

    # For Chinese and other CJK languages, count characters
    if language == "zh":
        return len(text)
    else:
        # For English and other space-separated languages, count words
        return len(text.split())


def tokenize_text(text: str, language: str = "en") -> List[str]:
    """
    Tokenize text into words with language-aware logic.

    For Chinese: Each character is a token
    For English: Space-separated tokenization

    Args:
        text: Text to tokenize
        language: Language code ('en' for English, 'zh' for Chinese, etc.)

    Returns:
        List of tokens

    Examples:
        >>> tokenize_text("Hello world", "en")
        ['Hello', 'world']
        >>> tokenize_text("你好世界", "zh")
        ['你', '好', '世', '界']
    """
    if not text:
        return []

    # Strip language parameter
    language = language.strip() if language else "en"

    # For Chinese and other CJK languages, split into characters
    if language == "zh":
        return list(text)
    else:
        # For English and other space-separated languages, split by spaces
        return text.split()
