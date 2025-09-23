"""
Statistical analyzer for extracting linguistic patterns and insights from influencer content
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Counter
from collections import Counter, defaultdict
from pathlib import Path

import spacy
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures, QuadgramAssocMeasures
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ..data.models.persona_constitution import StatisticalReport, CollocationItem
from ..config.settings import Settings
from ..utils.logging import get_logger
from ..utils.device_manager import get_device_manager
from .analysis_cache import AnalysisCacheManager


class StatisticalAnalyzer:
    """
    Analyzes text content to extract statistical patterns, linguistic features,
    and insights about speaking style and content themes.
    """
    
    def __init__(self, settings: Settings, persona_id: Optional[str] = None):
        """Initialize the statistical analyzer"""
        self.settings = settings
        self.persona_id = persona_id
        self.logger = get_logger(__name__)
        
        # Initialize spaCy model
        self.spacy_model = None
        self._init_spacy()
        
        # Initialize NLTK components
        self._init_nltk()
        
        # Report device usage for other libraries
        self._report_library_devices()
        
        # Initialize cache manager if persona_id provided
        self.cache_manager = None
        if persona_id:
            self.cache_manager = AnalysisCacheManager(settings, persona_id)
        
        # Analysis results storage (legacy in-memory cache)
        self.analysis_cache = {}
        
    def _init_spacy(self):
        """Initialize spaCy model with device reporting"""
        spacy_config = self.settings.statistical_analysis.spacy
        model_name = spacy_config.get('model', 'en_core_web_sm')
        
        # Get device manager for reporting
        device_manager = get_device_manager()
        
        try:
            self.spacy_model = spacy.load(model_name)
            
            # Configure model settings
            max_length = spacy_config.get('max_length', 1000000)
            self.spacy_model.max_length = max_length
            
            # Report device usage for spaCy
            # spaCy's standard models run on CPU, but transformers models can use GPU
            has_transformers = any('transformer' in pipe_name for pipe_name in self.spacy_model.pipe_names)
            
            if has_transformers and device_manager.is_gpu_available():
                device_manager.log_library_device_usage(
                    "spaCy", 
                    device_manager.get_torch_device().upper(), 
                    f"Model: {model_name} (transformers pipeline)"
                )
            else:
                device_manager.log_library_device_usage(
                    "spaCy", 
                    "CPU", 
                    f"Model: {model_name} (standard pipeline)"
                )
            
            self.logger.info(f"Loaded spaCy model: {model_name}")
            
        except OSError as e:
            self.logger.error(f"Failed to load spaCy model {model_name}: {e}")
            self.logger.info(f"Install with: python -m spacy download {model_name}")
            raise
    
    def _init_nltk(self):
        """Initialize NLTK components"""
        required_data = [
            'punkt',
            'stopwords', 
            'averaged_perceptron_tagger',
            'wordnet',
            'omw-1.4'
        ]
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                    self.logger.info(f"Downloaded NLTK data: {data_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK data {data_name}: {e}")
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
            self.logger.info("Initialized NLTK stopwords")
        except Exception as e:
            self.logger.warning(f"Failed to load stopwords: {e}")
            self.stop_words = set()
    
    def _report_library_devices(self):
        """Report device usage for statistical analysis libraries"""
        device_manager = get_device_manager()
        
        # NLTK always uses CPU
        device_manager.log_library_device_usage("NLTK", "CPU", "Text processing and tokenization")
        
        # scikit-learn uses CPU (no built-in GPU support)
        device_manager.log_library_device_usage("scikit-learn", "CPU", "TF-IDF and statistical analysis")
    
    def analyze_content(self, documents: List[Dict[str, Any]], 
                       use_cache: bool = True, 
                       force_reanalyze: bool = False,
                       max_cache_age_hours: Optional[int] = 24) -> StatisticalReport:
        """
        Perform comprehensive statistical analysis on the content
        
        Args:
            documents: List of document dictionaries with 'content' key
            use_cache: Whether to use cached analysis if available
            force_reanalyze: Force fresh analysis even if cache exists
            max_cache_age_hours: Maximum age of cache in hours (None = no limit)
            
        Returns:
            StatisticalReport with analysis results
        """
        # Try to load from cache first (if enabled and not forcing)
        if use_cache and not force_reanalyze and self.cache_manager:
            cached_report = self.cache_manager.load_analysis(documents, max_cache_age_hours)
            if cached_report:
                self.logger.info(f"Using cached statistical analysis for {len(documents)} documents")
                return cached_report
        
        # Perform fresh analysis
        self.logger.info(f"Starting {'fresh ' if force_reanalyze else ''}statistical analysis of {len(documents)} documents")
        
        # Combine all text content for most analyses
        all_text = " ".join([doc.get('content', '') for doc in documents])
        total_words = len(all_text.split())
        total_sentences = len(sent_tokenize(all_text))
        
        self.logger.info(f"Analyzing {total_words:,} words in {total_sentences:,} sentences")
        
        # Define analysis steps with progress tracking
        analysis_steps = [
            ("Extracting keywords", lambda: self._extract_keywords(documents)),
            ("Extracting entities", lambda: self._extract_entities(all_text)),
            ("Extracting collocations", lambda: self._extract_collocations(all_text)),
            ("Calculating readability", lambda: self._calculate_readability(all_text)),
            ("Analyzing linguistic patterns", lambda: self._analyze_linguistic_patterns(all_text)),
            ("Analyzing sentiment", lambda: self._analyze_sentiment(all_text))
        ]
        
        # Perform analysis with progress bar
        results = {}
        with tqdm(total=len(analysis_steps), desc="Statistical Analysis", unit="step") as pbar:
            for step_name, analysis_func in analysis_steps:
                pbar.set_description(f"Statistical Analysis: {step_name}")
                results[step_name.lower().replace(' ', '_')] = analysis_func()
                pbar.update(1)
        
        # Extract results
        keyword_analysis = results['extracting_keywords']
        entity_analysis = results['extracting_entities']
        collocation_analysis = results['extracting_collocations']
        readability_metrics = results['calculating_readability']
        linguistic_patterns = results['analyzing_linguistic_patterns']
        sentiment_analysis = results['analyzing_sentiment']
        
        # Create statistical report
        report = StatisticalReport(
            total_documents=len(documents),
            total_words=total_words,
            total_sentences=total_sentences,
            top_keywords=keyword_analysis,
            top_entities=entity_analysis,
            top_collocations=collocation_analysis,
            readability_metrics=readability_metrics,
            linguistic_patterns=linguistic_patterns,
            sentiment_analysis=sentiment_analysis
        )
        
        # Save to cache if cache manager available
        if self.cache_manager:
            try:
                self.cache_manager.save_analysis(report, documents)
                self.logger.debug("Saved analysis results to cache")
            except Exception as e:
                self.logger.warning(f"Failed to save analysis cache: {e}")
        
        self.logger.info("Statistical analysis completed")
        return report
    
    def has_cached_analysis(self, documents: List[Dict[str, Any]], 
                           max_age_hours: Optional[int] = 24) -> bool:
        """
        Check if cached analysis exists for the given documents
        
        Args:
            documents: Documents to check cache for
            max_age_hours: Maximum age in hours
            
        Returns:
            True if valid cache exists, False otherwise
        """
        if not self.cache_manager:
            return False
        return self.cache_manager.has_valid_cache(documents, max_age_hours)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached analyses
        
        Returns:
            Dictionary with cache information
        """
        if not self.cache_manager:
            return {"status": "no_cache_manager", "persona_id": self.persona_id}
        return self.cache_manager.get_cache_info()
    
    def clear_analysis_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached analysis files
        
        Args:
            older_than_days: Only clear files older than this many days (None = all)
        """
        if self.cache_manager:
            self.cache_manager.clear_cache(older_than_days)
        else:
            self.logger.warning("No cache manager available")
    
    def _extract_keywords(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Extract top keywords using TF-IDF from individual documents"""
        self.logger.debug("Extracting keywords from individual documents")
        
        keywords_config = self.settings.statistical_analysis.keywords
        max_features = keywords_config.get('max_features', 1000)
        min_frequency = keywords_config.get('min_frequency', 2)
        ngram_range = tuple(keywords_config.get('ngram_range', [1, 3]))
        
        # Extract document contents
        document_texts = [doc.get('content', '') for doc in documents if doc.get('content', '').strip()]
        
        if len(document_texts) == 0:
            self.logger.warning("No document content found for keyword extraction")
            return {}
        
        # Adjust min_df if we have fewer documents than the minimum frequency
        actual_min_df = min(min_frequency, len(document_texts))
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=actual_min_df,
            max_df=0.95,  # Ignore terms that appear in >95% of documents
            ngram_range=ngram_range,
            stop_words=list(self.stop_words),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        try:
            # Fit and transform on individual documents
            tfidf_matrix = vectorizer.fit_transform(document_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores across all documents
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create keyword dictionary
            keyword_scores = dict(zip(feature_names, mean_scores))
            
            # Sort by score and return top keywords
            sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Convert to frequency-like scores (multiply by 1000 and round)
            keywords = {keyword: int(score * 1000) for keyword, score in sorted_keywords[:100]}
            
            self.logger.debug(f"Extracted {len(keywords)} keywords from {len(document_texts)} documents")
            return keywords
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return {}
    
    def _extract_entities(self, text: str) -> Dict[str, int]:
        """Extract named entities using spaCy"""
        self.logger.debug("Extracting named entities")
        
        if not self.spacy_model:
            return {}
        
        entities = Counter()
        
        try:
            # Process text in chunks to handle large documents
            chunk_size = 100000  # 100k characters per chunk
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in tqdm(chunks, desc="Processing text chunks for entities", leave=False):
                doc = self.spacy_model(chunk)
                
                for ent in doc.ents:
                    # Filter out common entity types we don't want
                    if ent.label_ in ['CARDINAL', 'ORDINAL', 'QUANTITY', 'PERCENT', 'TIME', 'DATE']:
                        continue
                    
                    # Clean entity text
                    entity_text = ent.text.strip().lower()
                    if len(entity_text) > 2 and entity_text.isalpha():
                        entities[f"{entity_text} ({ent.label_})"] += 1
            
            # Return top entities
            top_entities = dict(entities.most_common(50))
            
            self.logger.debug(f"Extracted {len(top_entities)} entities")
            return top_entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return {}
    
    def _extract_collocations(self, text: str) -> List[Dict[str, Any]]:
        """Extract collocations (n-grams) using NLTK"""
        self.logger.debug("Extracting collocations")
        
        nltk_config = self.settings.statistical_analysis.nltk
        window_size = nltk_config.get('collocations', {}).get('window_size', 2)
        min_frequency = nltk_config.get('collocations', {}).get('min_frequency', 3)
        significance_threshold = nltk_config.get('collocations', {}).get('significance_threshold', 5.0)
        
        try:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            
            # Filter tokens
            filtered_tokens = [
                token for token in tokens 
                if token.isalpha() 
                and len(token) > 2 
                and token not in self.stop_words
            ]
            
            collocations = []
            
            # Extract bigrams
            bigram_finder = BigramCollocationFinder.from_words(filtered_tokens, window_size=window_size)
            bigram_finder.apply_freq_filter(min_frequency)
            
            bigram_measures = BigramAssocMeasures()
            top_bigrams = bigram_finder.nbest(bigram_measures.chi_sq, 25)
            
            for bigram in top_bigrams:
                score = bigram_finder.score_ngram(bigram_measures.chi_sq, bigram[0], bigram[1])
                if score >= significance_threshold:
                    collocations.append({
                        'ngram': ' '.join(bigram),
                        'type': 'bigram',
                        'frequency': bigram_finder.ngram_fd[bigram],
                        'score': round(score, 2)
                    })
            
            # Extract trigrams
            trigram_finder = TrigramCollocationFinder.from_words(filtered_tokens, window_size=window_size)
            trigram_finder.apply_freq_filter(min_frequency)
            
            trigram_measures = TrigramAssocMeasures()
            top_trigrams = trigram_finder.nbest(trigram_measures.chi_sq, 20)
            
            for trigram in top_trigrams:
                try:
                    score = trigram_finder.score_ngram(trigram_measures.chi_sq, *trigram)
                    if score >= significance_threshold:
                        collocations.append({
                            'ngram': ' '.join(trigram),
                            'type': 'trigram', 
                            'frequency': trigram_finder.ngram_fd[trigram],
                            'score': round(score, 2)
                        })
                except:
                    continue
            
            # Sort by score
            collocations.sort(key=lambda x: x['score'], reverse=True)
            
            # Convert to CollocationItem objects
            collocation_items = [
                CollocationItem(**collocation_dict) 
                for collocation_dict in collocations[:50]  # Top 50
            ]
            
            self.logger.debug(f"Extracted {len(collocation_items)} collocations")
            return collocation_items
            
        except Exception as e:
            self.logger.error(f"Collocation extraction failed: {e}")
            return []
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        self.logger.debug("Calculating readability metrics")
        
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Filter to actual words
            word_tokens = [word for word in words if word.isalpha()]
            
            if not sentences or not word_tokens:
                return {}
            
            # Basic metrics
            avg_sentence_length = len(word_tokens) / len(sentences)
            avg_word_length = sum(len(word) for word in word_tokens) / len(word_tokens)
            
            # Syllable estimation (simple heuristic)
            def count_syllables(word):
                word = word.lower()
                vowels = 'aeiouy'
                syllable_count = 0
                previous_was_vowel = False
                
                for char in word:
                    if char in vowels:
                        if not previous_was_vowel:
                            syllable_count += 1
                        previous_was_vowel = True
                    else:
                        previous_was_vowel = False
                
                # Handle silent e
                if word.endswith('e') and syllable_count > 1:
                    syllable_count -= 1
                
                return max(1, syllable_count)
            
            total_syllables = sum(count_syllables(word) for word in word_tokens)
            avg_syllables_per_word = total_syllables / len(word_tokens)
            
            # Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
            
            # Flesch-Kincaid Grade Level
            grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            grade_level = max(0, grade_level)
            
            metrics = {
                'avg_sentence_length': round(avg_sentence_length, 2),
                'avg_word_length': round(avg_word_length, 2),
                'avg_syllables_per_word': round(avg_syllables_per_word, 2),
                'flesch_reading_ease': round(flesch_score, 2),
                'flesch_kincaid_grade': round(grade_level, 2),
                'total_sentences': len(sentences),
                'total_words': len(word_tokens),
                'total_syllables': total_syllables
            }
            
            self.logger.debug("Readability metrics calculated")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Readability calculation failed: {e}")
            return {}
    
    def _analyze_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns and style"""
        self.logger.debug("Analyzing linguistic patterns")
        
        try:
            patterns = {}
            
            # Sentence patterns
            sentences = sent_tokenize(text)
            
            # Question frequency
            questions = [s for s in sentences if s.strip().endswith('?')]
            patterns['question_ratio'] = len(questions) / len(sentences) if sentences else 0
            
            # Exclamation frequency
            exclamations = [s for s in sentences if s.strip().endswith('!')]
            patterns['exclamation_ratio'] = len(exclamations) / len(sentences) if sentences else 0
            
            # Sentence length distribution
            sentence_lengths = [len(s.split()) for s in sentences]
            if sentence_lengths:
                patterns['sentence_length_stats'] = {
                    'min': min(sentence_lengths),
                    'max': max(sentence_lengths),
                    'avg': sum(sentence_lengths) / len(sentence_lengths),
                    'median': sorted(sentence_lengths)[len(sentence_lengths) // 2]
                }
            
            # POS tag patterns using NLTK
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            
            pos_counts = Counter(tag for word, tag in pos_tags)
            total_tags = len(pos_tags)
            
            if total_tags > 0:
                patterns['pos_distribution'] = {
                    tag: count / total_tags 
                    for tag, count in pos_counts.most_common(10)
                }
            
            # Punctuation patterns
            punctuation_chars = '.,!?;:'
            punct_counts = Counter(char for char in text if char in punctuation_chars)
            total_punct = sum(punct_counts.values())
            
            if total_punct > 0:
                patterns['punctuation_distribution'] = dict(punct_counts)
            
            # Word frequency patterns
            word_tokens = [word.lower() for word in words if word.isalpha()]
            word_freq = Counter(word_tokens)
            
            patterns['vocabulary_richness'] = len(set(word_tokens)) / len(word_tokens) if word_tokens else 0
            patterns['most_common_words'] = dict(word_freq.most_common(20))
            
            self.logger.debug("Linguistic pattern analysis completed")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Linguistic pattern analysis failed: {e}")
            return {}
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis"""
        self.logger.debug("Analyzing sentiment")
        
        try:
            # Simple sentiment analysis using word lists
            positive_words = {
                'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome',
                'love', 'perfect', 'brilliant', 'outstanding', 'superb', 'magnificent',
                'good', 'better', 'best', 'strong', 'powerful', 'effective',
                'success', 'successful', 'win', 'winning', 'achieve', 'achievement'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
                'fail', 'failure', 'wrong', 'problem', 'issue', 'difficult',
                'hard', 'struggle', 'pain', 'hurt', 'damage', 'broken',
                'weak', 'poor', 'worse', 'worst', 'lose', 'losing'
            }
            
            words = word_tokenize(text.lower())
            word_tokens = [word for word in words if word.isalpha()]
            
            positive_count = sum(1 for word in word_tokens if word in positive_words)
            negative_count = sum(1 for word in word_tokens if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words > 0:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
            else:
                sentiment_score = 0.0
            
            sentiment = {
                'overall_sentiment': sentiment_score,
                'positive_ratio': positive_count / len(word_tokens) if word_tokens else 0,
                'negative_ratio': negative_count / len(word_tokens) if word_tokens else 0,
                'positive_words_count': positive_count,
                'negative_words_count': negative_count,
                'total_words': len(word_tokens)
            }
            
            self.logger.debug("Sentiment analysis completed")
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {}
    
    def extract_signature_phrases(self, text: str, min_length: int = 3, max_length: int = 8) -> List[str]:
        """Extract potential signature phrases and catchphrases"""
        self.logger.debug("Extracting signature phrases")
        
        try:
            sentences = sent_tokenize(text)
            phrases = []
            
            # Look for repeated phrases
            phrase_counts = Counter()
            
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                
                # Extract n-grams of different lengths
                for n in range(min_length, max_length + 1):
                    for ngram in ngrams(words, n):
                        phrase = ' '.join(ngram)
                        
                        # Filter out phrases with stopwords only
                        if not all(word in self.stop_words for word in ngram):
                            phrase_counts[phrase] += 1
            
            # Get phrases that appear multiple times
            repeated_phrases = [
                phrase for phrase, count in phrase_counts.items() 
                if count >= 2
            ]
            
            # Look for phrases with specific patterns
            pattern_phrases = []
            
            # Common catchphrase patterns
            patterns = [
                r'\b(listen|look|here\'s|now|so|but|and)\s+[^.!?]*',
                r'\bthe\s+(point|thing|reality|truth)\s+is\b[^.!?]*',
                r'\byou\s+(need|have|want|should|must)\s+to\b[^.!?]*',
                r'\blet\s+me\s+(tell|show|explain)\b[^.!?]*'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                pattern_phrases.extend(matches[:5])  # Limit to 5 per pattern
            
            # Combine and deduplicate
            all_phrases = list(set(repeated_phrases + pattern_phrases))
            
            # Filter and clean
            filtered_phrases = []
            for phrase in all_phrases:
                phrase = phrase.strip()
                if (len(phrase) > 10 and 
                    len(phrase) < 100 and 
                    not phrase.startswith(('the', 'and', 'or', 'but'))):
                    filtered_phrases.append(phrase)
            
            self.logger.debug(f"Extracted {len(filtered_phrases)} signature phrases")
            return filtered_phrases[:20]  # Return top 20
            
        except Exception as e:
            self.logger.error(f"Signature phrase extraction failed: {e}")
            return []
    
    def get_vocabulary_insights(self, text: str) -> Dict[str, Any]:
        """Get insights about vocabulary usage"""
        self.logger.debug("Analyzing vocabulary insights")
        
        try:
            words = word_tokenize(text.lower())
            word_tokens = [word for word in words if word.isalpha() and len(word) > 2]
            
            # Frequency analysis
            word_freq = Counter(word_tokens)
            
            # Filter out stop words for content words
            content_words = [word for word in word_tokens if word not in self.stop_words]
            content_freq = Counter(content_words)
            
            # Calculate insights
            insights = {
                'total_words': len(word_tokens),
                'unique_words': len(set(word_tokens)),
                'vocabulary_richness': len(set(word_tokens)) / len(word_tokens) if word_tokens else 0,
                'content_words': len(content_words),
                'unique_content_words': len(set(content_words)),
                'top_content_words': dict(content_freq.most_common(30)),
                'word_length_distribution': {
                    'avg_length': sum(len(word) for word in word_tokens) / len(word_tokens) if word_tokens else 0,
                    'length_counts': dict(Counter(len(word) for word in word_tokens))
                }
            }
            
            self.logger.debug("Vocabulary insights analysis completed")
            return insights
            
        except Exception as e:
            self.logger.error(f"Vocabulary insights analysis failed: {e}")
            return {}