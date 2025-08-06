import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List

class EmotionAnalyzer:
    """
    Emotion analyzer using EmoBank methodology for calculating valence and arousal scores.
    """
    
    def __init__(self):
        """Initialize the emotion analyzer with EmoBank lexicon."""
        self.lexicon = self._load_emobank_lexicon()
        
    def _load_emobank_lexicon(self) -> Dict[str, Tuple[float, float]]:
        """
        Load EmoBank lexicon data.
        Returns dictionary mapping words to (valence, arousal) tuples.
        """
        # Try to load the full EmoBank dataset first
        full_emobank_path = "data/emobank_full.csv"
        lexicon_path = "data/emobank_lexicon.csv"
        
        try:
            # First try to use the full EmoBank dataset
            if os.path.exists(full_emobank_path):
                df = pd.read_csv(full_emobank_path)
                lexicon = self._create_lexicon_from_emobank(df)
                if lexicon:
                    return lexicon
            
            # Fall back to the existing lexicon CSV
            if os.path.exists(lexicon_path):
                df = pd.read_csv(lexicon_path)
                lexicon = {}
                for _, row in df.iterrows():
                    word = str(row['word']).lower().strip()
                    valence = float(row['valence'])
                    arousal = float(row['arousal'])
                    lexicon[word] = (valence, arousal)
                return lexicon
            else:
                # Create a basic emotion lexicon based on EmoBank methodology
                return self._create_basic_lexicon()
                
        except Exception as e:
            print(f"Error loading EmoBank lexicon: {e}")
            return self._create_basic_lexicon()
    
    def _create_lexicon_from_emobank(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Create a word-level lexicon from the EmoBank dataset.
        
        Args:
            df: EmoBank dataframe with columns: id, split, V, A, D, text
            
        Returns:
            Dictionary mapping words to (valence, arousal) tuples
        """
        import re
        from collections import defaultdict
        
        try:
            # Create word-level averages from sentence-level ratings
            word_emotions = defaultdict(list)
            
            for _, row in df.iterrows():
                try:
                    text = str(row['text']).lower()
                    valence = float(row['V'])  # V column is valence
                    arousal = float(row['A'])  # A column is arousal
                    
                    # Skip invalid ratings
                    if pd.isna(valence) or pd.isna(arousal):
                        continue
                    
                    # Convert from 1-5 scale to 1-9 scale to match our existing system
                    valence_scaled = ((valence - 1) / 4) * 8 + 1  # Convert 1-5 to 1-9
                    arousal_scaled = ((arousal - 1) / 4) * 8 + 1  # Convert 1-5 to 1-9
                    
                    # Extract words from text
                    words = re.findall(r'\b[a-zA-Z]+\b', text)
                    
                    for word in words:
                        word = word.lower().strip()
                        if len(word) >= 3:  # Only include words with 3+ characters
                            word_emotions[word].append((valence_scaled, arousal_scaled))
                
                except (ValueError, KeyError):
                    continue
            
            # Average the emotions for each word
            lexicon = {}
            for word, emotions in word_emotions.items():
                if len(emotions) >= 2:  # Only include words that appear at least twice
                    avg_valence = np.mean([e[0] for e in emotions])
                    avg_arousal = np.mean([e[1] for e in emotions])
                    lexicon[word] = (avg_valence, avg_arousal)
            
            print(f"Created lexicon from EmoBank data with {len(lexicon)} words")
            return lexicon
            
        except Exception as e:
            print(f"Error creating lexicon from EmoBank: {e}")
            return {}
    
    def _create_basic_lexicon(self) -> Dict[str, Tuple[float, float]]:
        """
        Create a basic emotion lexicon following EmoBank methodology.
        Values are on a 1-9 scale where:
        - Valence: 1 (very negative) to 9 (very positive)
        - Arousal: 1 (very calm) to 9 (very excited)
        """
        # Basic lexicon with words commonly found in scam messages
        basic_words = {
            # High arousal, negative valence (typical scam urgency words)
            'urgent': (2.0, 8.0), 'immediately': (3.0, 7.5), 'emergency': (2.5, 8.5),
            'warning': (2.5, 8.0), 'alert': (3.0, 8.0), 'danger': (1.5, 8.5),
            'threat': (1.5, 8.0), 'risk': (2.5, 7.0), 'expire': (2.0, 7.5),
            'deadline': (3.0, 7.0), 'limited': (3.5, 6.5), 'hurry': (4.0, 8.0),
            
            # High arousal, positive valence (typical scam reward words)
            'win': (8.0, 7.5), 'winner': (8.5, 8.0), 'prize': (8.0, 7.0),
            'reward': (8.0, 6.5), 'bonus': (7.5, 6.0), 'gift': (8.0, 6.5),
            'free': (7.0, 6.0), 'congratulations': (8.5, 7.5), 'selected': (7.0, 6.0),
            'lucky': (8.0, 7.0), 'exclusive': (7.5, 6.5), 'special': (7.0, 5.5),
            
            # Money-related words (varied valence, moderate to high arousal)
            'money': (6.5, 6.0), 'cash': (6.5, 6.5), 'dollar': (6.0, 5.5),
            'profit': (7.5, 6.5), 'income': (7.0, 5.0), 'rich': (8.0, 6.5),
            'million': (7.5, 7.5), 'thousand': (7.0, 6.0), 'investment': (6.5, 5.5),
            'return': (6.0, 5.0), 'guarantee': (7.0, 5.5), 'earn': (7.0, 6.0),
            
            # Action words (moderate valence, high arousal)
            'click': (5.0, 6.5), 'call': (5.0, 6.0), 'verify': (4.5, 6.5),
            'confirm': (5.5, 6.0), 'update': (5.5, 5.5), 'activate': (6.0, 6.5),
            'claim': (6.5, 7.0), 'redeem': (7.0, 6.5), 'download': (5.5, 5.5),
            'install': (5.0, 5.5), 'register': (5.0, 5.5), 'subscribe': (5.0, 5.0),
            
            # Security/trust words (varied based on context)
            'secure': (7.0, 4.0), 'safe': (7.5, 3.5), 'protect': (7.0, 5.5),
            'account': (5.0, 5.0), 'bank': (5.5, 5.5), 'credit': (6.0, 5.5),
            'card': (5.0, 5.5), 'password': (4.0, 6.0), 'login': (5.0, 5.5),
            'security': (6.0, 6.0), 'fraud': (2.0, 7.5), 'scam': (1.5, 7.0),
            
            # Emotional manipulation words
            'help': (7.0, 5.5), 'assist': (7.0, 5.0), 'support': (7.5, 5.0),
            'trust': (8.0, 4.5), 'honest': (8.0, 4.0), 'legitimate': (7.5, 4.0),
            'real': (6.5, 5.0), 'genuine': (7.5, 4.5), 'authentic': (7.5, 4.5),
            'official': (7.0, 5.0), 'authorized': (7.0, 5.5), 'verified': (7.5, 5.5),
            
            # Negative consequence words
            'suspended': (2.0, 7.0), 'blocked': (2.5, 7.5), 'closed': (3.0, 6.5),
            'terminated': (2.0, 7.5), 'cancelled': (3.0, 6.5), 'frozen': (2.5, 7.0),
            'penalty': (2.0, 7.0), 'fee': (3.5, 5.5), 'charge': (3.5, 6.0),
            'debt': (2.0, 6.5), 'owe': (2.5, 6.0), 'overdue': (2.5, 7.0),
            
            # Common positive words
            'good': (7.0, 4.5), 'great': (8.0, 6.0), 'excellent': (8.5, 6.5),
            'amazing': (8.5, 7.5), 'fantastic': (8.5, 7.5), 'wonderful': (8.5, 6.5),
            'perfect': (8.5, 6.0), 'best': (8.0, 6.5), 'top': (7.5, 6.0),
            'success': (8.0, 7.0), 'opportunity': (7.5, 6.5), 'chance': (6.5, 6.0),
            
            # Common negative words
            'bad': (2.5, 5.5), 'terrible': (1.5, 7.0), 'awful': (1.5, 7.0),
            'horrible': (1.5, 7.5), 'worst': (1.5, 7.0), 'fail': (2.0, 6.5),
            'problem': (2.5, 6.5), 'issue': (3.0, 6.0), 'error': (2.5, 6.5),
            'mistake': (2.5, 5.5), 'wrong': (2.5, 6.0), 'trouble': (2.5, 7.0),
            
            # Neutral/common words
            'the': (5.0, 1.0), 'and': (5.0, 1.0), 'or': (5.0, 1.0),
            'but': (5.0, 2.0), 'if': (5.0, 3.0), 'then': (5.0, 2.5),
            'you': (5.0, 2.0), 'your': (5.0, 2.0), 'we': (5.0, 2.0),
            'our': (5.0, 2.0), 'this': (5.0, 2.0), 'that': (5.0, 2.0),
            'have': (5.0, 2.5), 'has': (5.0, 2.5), 'will': (5.0, 3.0),
            'can': (6.0, 3.0), 'may': (5.0, 2.5), 'must': (5.0, 5.0),
            'should': (5.0, 4.0), 'need': (4.5, 5.0), 'want': (6.0, 4.5),
            'time': (5.0, 4.0), 'day': (5.0, 3.0), 'week': (5.0, 3.0),
            'month': (5.0, 3.0), 'year': (5.0, 3.0), 'today': (5.5, 4.5),
            'now': (5.0, 6.0), 'soon': (5.5, 5.5), 'before': (5.0, 4.0),
            'after': (5.0, 3.5), 'new': (6.5, 5.0), 'old': (4.0, 3.0),
            'first': (6.0, 4.5), 'last': (4.5, 4.5), 'next': (5.5, 4.5),
            'many': (5.0, 3.0), 'much': (5.0, 3.5), 'more': (6.0, 4.0),
            'most': (6.5, 4.0), 'all': (5.5, 3.5), 'some': (5.0, 3.0),
            'every': (5.5, 4.0), 'any': (5.0, 3.5), 'no': (3.5, 4.5),
            'not': (3.5, 4.0), 'only': (5.0, 4.0), 'also': (5.5, 3.5),
            'get': (6.0, 5.0), 'go': (5.5, 5.5), 'come': (5.5, 4.5),
            'make': (6.0, 5.0), 'take': (5.5, 4.5), 'give': (7.0, 4.5),
            'find': (6.0, 5.0), 'know': (6.5, 4.0), 'think': (5.5, 4.0),
            'see': (5.5, 4.5), 'look': (5.5, 4.5), 'use': (5.5, 4.0),
            'work': (6.0, 5.5), 'way': (5.5, 4.0), 'place': (5.0, 3.5),
            'right': (7.0, 4.5), 'left': (5.0, 3.5), 'here': (5.5, 4.0),
            'there': (5.0, 3.5), 'where': (5.0, 4.5), 'when': (5.0, 4.5),
            'why': (5.0, 5.0), 'how': (5.5, 4.5), 'what': (5.0, 4.5),
            'who': (5.0, 4.0), 'which': (5.0, 4.0), 'about': (5.0, 3.5),
            'from': (5.0, 3.0), 'with': (5.5, 3.5), 'without': (4.0, 4.5),
            'through': (5.0, 4.0), 'between': (5.0, 3.5), 'over': (5.5, 4.0),
            'under': (4.5, 4.0), 'into': (5.5, 4.5), 'onto': (5.5, 4.5),
        }
        
        return basic_words
    
    def analyze_emotion(self, text: str) -> Tuple[float, float, int, float]:
        """
        Analyze emotion in text using EmoBank methodology.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (valence, arousal, word_count, coverage)
            - valence: 1-9 scale (1=very negative, 9=very positive)
            - arousal: 1-9 scale (1=very calm, 9=very excited)  
            - word_count: Number of words processed
            - coverage: Percentage of words found in lexicon
        """
        if not text or not isinstance(text, str):
            return np.nan, np.nan, 0, 0.0
        
        words = text.lower().split()
        if not words:
            return np.nan, np.nan, 0, 0.0
        
        valences = []
        arousals = []
        found_words = 0
        
        for word in words:
            # Remove common punctuation
            clean_word = word.strip('.,!?";:()[]{}')
            
            if clean_word in self.lexicon:
                valence, arousal = self.lexicon[clean_word]
                valences.append(valence)
                arousals.append(arousal)
                found_words += 1
        
        if not valences:
            return np.nan, np.nan, len(words), 0.0
        
        # Calculate mean scores
        mean_valence = float(np.mean(valences))
        mean_arousal = float(np.mean(arousals))
        
        # Calculate coverage
        coverage = found_words / len(words) if words else 0.0
        
        return mean_valence, mean_arousal, len(words), coverage
    
    def get_emotion_category(self, valence: float, arousal: float) -> str:
        """
        Categorize emotion based on valence and arousal scores.
        
        Args:
            valence: Valence score (1-9)
            arousal: Arousal score (1-9)
            
        Returns:
            Emotion category string
        """
        if np.isnan(valence) or np.isnan(arousal):
            return "Unknown"
        
        # Define quadrants based on midpoint (5.0)
        high_valence = valence > 5.0
        high_arousal = arousal > 5.0
        
        if high_valence and high_arousal:
            if valence > 7.0 and arousal > 7.0:
                return "Highly Positive & Exciting"
            else:
                return "Positive & Energetic"
        elif not high_valence and high_arousal:
            if valence < 3.0 and arousal > 7.0:
                return "Highly Negative & Alarming"
            else:
                return "Negative & Intense"
        elif high_valence and not high_arousal:
            return "Positive & Calm"
        else:
            if valence < 3.0 and arousal < 3.0:
                return "Highly Negative & Low Energy"
            else:
                return "Negative & Low Energy"
    
    def get_lexicon_stats(self) -> Dict[str, any]:
        """Get statistics about the loaded lexicon."""
        if not self.lexicon:
            return {"size": 0, "valence_range": (0, 0), "arousal_range": (0, 0)}
        
        valences = [v[0] for v in self.lexicon.values()]
        arousals = [v[1] for v in self.lexicon.values()]
        
        return {
            "size": len(self.lexicon),
            "valence_range": (min(valences), max(valences)),
            "arousal_range": (min(arousals), max(arousals)),
            "mean_valence": np.mean(valences),
            "mean_arousal": np.mean(arousals)
        }
