import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class TextProcessor:
    """
    Text preprocessing utility for emotion analysis.
    """
    
    def __init__(self):
        """Initialize the text processor with required NLTK data."""
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = self._get_stopwords()
    
    def _download_nltk_data(self):
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def _get_stopwords(self) -> Set[str]:
        """Get English stopwords set."""
        try:
            return set(stopwords.words('english'))
        except Exception:
            # Fallback stopwords if NLTK download fails
            return {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                'with', 'through', 'during', 'before', 'after', 'above', 'below', 
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
    
    def preprocess_text(self, text: str, 
                       min_word_length: int = 3,
                       remove_stopwords: bool = True,
                       apply_stemming: bool = False,
                       remove_numbers: bool = True,
                       remove_urls: bool = True) -> str:
        """
        Preprocess text for emotion analysis.
        
        Args:
            text: Input text to preprocess
            min_word_length: Minimum word length to keep
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming
            remove_numbers: Whether to remove numbers
            remove_urls: Whether to remove URLs
            
        Returns:
            Preprocessed text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '', text)
        
        # Remove numbers if requested
        if remove_numbers:
            text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback tokenization if NLTK fails
            tokens = text.split()
        
        # Remove punctuation and filter tokens
        processed_tokens = []
        for token in tokens:
            # Remove punctuation
            token = token.translate(str.maketrans('', '', string.punctuation))
            
            # Skip if empty after punctuation removal
            if not token:
                continue
            
            # Skip if too short
            if len(token) < min_word_length:
                continue
            
            # Skip if stopword (if requested)
            if remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming if requested
            if apply_stemming:
                try:
                    token = self.stemmer.stem(token)
                except Exception:
                    pass  # Keep original token if stemming fails
            
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def extract_features(self, text: str) -> dict:
        """
        Extract text features for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        if not text or not isinstance(text, str):
            return {
                'length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'caps_ratio': 0,
                'number_count': 0,
                'punctuation_ratio': 0
            }
        
        # Basic counts
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Sentence count (approximate)
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count == 0:
            sentence_count = 1  # At least one sentence
        
        # Average word length
        avg_word_length = sum(len(word.strip(string.punctuation)) for word in words) / word_count if word_count > 0 else 0
        
        # Punctuation counts
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Capitalization ratio
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / length if length > 0 else 0
        
        # Number count
        number_count = len(re.findall(r'\b\d+\b', text))
        
        # Punctuation ratio
        punctuation_count = sum(1 for c in text if c in string.punctuation)
        punctuation_ratio = punctuation_count / length if length > 0 else 0
        
        return {
            'length': length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_ratio': round(caps_ratio, 3),
            'number_count': number_count,
            'punctuation_ratio': round(punctuation_ratio, 3)
        }
    
    def is_likely_scam_pattern(self, text: str) -> dict:
        """
        Detect common scam message patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with pattern detection results
        """
        if not text or not isinstance(text, str):
            return {'is_suspicious': False, 'patterns': [], 'confidence': 0.0}
        
        text_lower = text.lower()
        patterns = []
        
        # Urgency patterns
        urgency_words = ['urgent', 'immediate', 'asap', 'quickly', 'hurry', 'deadline', 'expire', 'limited time']
        if any(word in text_lower for word in urgency_words):
            patterns.append('urgency')
        
        # Money/reward patterns
        money_patterns = [
            r'\$\d+', r'\d+\s*dollars?', r'\d+\s*usd', 'million', 'thousand',
            'prize', 'reward', 'win', 'winner', 'won', 'free money', 'cash'
        ]
        if any(re.search(pattern, text_lower) for pattern in money_patterns):
            patterns.append('money_reward')
        
        # Action request patterns
        action_words = ['click here', 'call now', 'verify', 'confirm', 'update', 'claim', 'download']
        if any(phrase in text_lower for phrase in action_words):
            patterns.append('action_request')
        
        # Threat patterns
        threat_words = ['suspended', 'blocked', 'terminated', 'penalty', 'legal action', 'arrest']
        if any(word in text_lower for word in threat_words):
            patterns.append('threat')
        
        # Personal info request
        personal_info = ['ssn', 'social security', 'credit card', 'password', 'pin', 'account number']
        if any(info in text_lower for info in personal_info):
            patterns.append('personal_info_request')
        
        # Excessive punctuation/capitalization
        if text.count('!') > 2 or text.count('?') > 2:
            patterns.append('excessive_punctuation')
        
        if sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            patterns.append('excessive_caps')
        
        # Calculate confidence based on number of patterns
        confidence = min(len(patterns) / 3.0, 1.0)  # Max confidence of 1.0
        is_suspicious = len(patterns) >= 2  # Suspicious if 2+ patterns
        
        return {
            'is_suspicious': is_suspicious,
            'patterns': patterns,
            'confidence': round(confidence, 2)
        }
    
    def clean_for_display(self, text: str, max_length: int = 100) -> str:
        """
        Clean text for display purposes.
        
        Args:
            text: Input text
            max_length: Maximum length for display
            
        Returns:
            Cleaned text suitable for display
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
        
        return text
