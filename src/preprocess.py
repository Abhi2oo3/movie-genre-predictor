import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from typing import List, Tuple
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    A class for preprocessing text data for movie genre prediction.
    """
    
    def __init__(self, use_stemming=True, use_lemmatization=False):
        """
        Initialize the text preprocessor.
        
        Args:
            use_stemming (bool): Whether to use stemming
            use_lemmatization (bool): Whether to use lemmatization
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words specific to movie plots
        custom_stops = {
            'movie', 'film', 'story', 'plot', 'character', 'characters',
            'scene', 'scenes', 'director', 'actor', 'actress', 'cast',
            'year', 'years', 'time', 'day', 'night', 'man', 'woman',
            'boy', 'girl', 'child', 'children', 'family', 'friend', 'friends'
        }
        self.stop_words.update(custom_stops)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokenized text.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Stemmed tokens
        """
        if self.stemmer:
            return [self.stemmer.stem(token) for token in tokens]
        return tokens
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = self.apply_stemming(tokens)
        elif self.use_lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess text data in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column to preprocess
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        df_copy = df.copy()
        df_copy[f'{text_column}_processed'] = df_copy[text_column].apply(self.preprocess_text)
        return df_copy

def extract_genres(genre_string: str) -> List[str]:
    """
    Extract individual genres from a comma-separated genre string.
    
    Args:
        genre_string (str): Comma-separated genre string
        
    Returns:
        List[str]: List of individual genres
    """
    if pd.isna(genre_string) or genre_string == '':
        return []
    
    # Split by comma and clean each genre
    genres = [genre.strip() for genre in genre_string.split(',')]
    return [genre for genre in genres if genre]

def get_all_genres(df: pd.DataFrame, genre_column: str) -> List[str]:
    """
    Get all unique genres from the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        genre_column (str): Name of the genre column
        
    Returns:
        List[str]: List of all unique genres
    """
    all_genres = []
    for genres in df[genre_column].apply(extract_genres):
        all_genres.extend(genres)
    
    return sorted(list(set(all_genres)))

def create_genre_columns(df: pd.DataFrame, genre_column: str) -> pd.DataFrame:
    """
    Create binary columns for each genre.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        genre_column (str): Name of the genre column
        
    Returns:
        pd.DataFrame: DataFrame with binary genre columns
    """
    df_copy = df.copy()
    all_genres = get_all_genres(df, genre_column)
    
    for genre in all_genres:
        df_copy[f'genre_{genre.lower().replace(" ", "_")}'] = df_copy[genre_column].apply(
            lambda x: 1 if genre in extract_genres(x) else 0
        )
    
    return df_copy

def get_most_common_words_by_genre(df: pd.DataFrame, text_column: str, genre_column: str, 
                                  top_n: int = 10) -> dict:
    """
    Get the most common words for each genre.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        genre_column (str): Name of the genre column
        top_n (int): Number of top words to return per genre
        
    Returns:
        dict: Dictionary with genres as keys and lists of (word, count) tuples as values
    """
    from collections import Counter
    
    genre_words = {}
    all_genres = get_all_genres(df, genre_column)
    
    for genre in all_genres:
        # Get all plots for this genre
        genre_plots = df[df[genre_column].str.contains(genre, na=False)][text_column]
        
        # Combine all words
        all_words = []
        for plot in genre_plots:
            if pd.notna(plot):
                words = plot.lower().split()
                all_words.extend(words)
        
        # Count words and get top N
        word_counts = Counter(all_words)
        genre_words[genre] = word_counts.most_common(top_n)
    
    return genre_words

def analyze_text_statistics(df: pd.DataFrame, text_column: str) -> dict:
    """
    Analyze basic text statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        
    Returns:
        dict: Dictionary with text statistics
    """
    stats = {}
    
    # Word count statistics
    word_counts = df[text_column].str.split().str.len()
    stats['avg_words_per_plot'] = word_counts.mean()
    stats['min_words_per_plot'] = word_counts.min()
    stats['max_words_per_plot'] = word_counts.max()
    stats['std_words_per_plot'] = word_counts.std()
    
    # Character count statistics
    char_counts = df[text_column].str.len()
    stats['avg_chars_per_plot'] = char_counts.mean()
    stats['min_chars_per_plot'] = char_counts.min()
    stats['max_chars_per_plot'] = char_counts.max()
    stats['std_chars_per_plot'] = char_counts.std()
    
    return stats 