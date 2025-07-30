import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import joblib
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class MovieGenrePredictor:
    """
    A class for training and using movie genre prediction models.
    """
    
    def __init__(self, vectorizer_type='tfidf', model_type='logistic'):
        """
        Initialize the movie genre predictor.
        
        Args:
            vectorizer_type (str): Type of vectorizer ('count' or 'tfidf')
            model_type (str): Type of model ('logistic', 'naive_bayes', or 'random_forest')
        """
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.genre_columns = None
        self.is_trained = False
        
    def _create_vectorizer(self, max_features=5000, ngram_range=(1, 2)):
        """
        Create the text vectorizer.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to consider
        """
        if self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
    
    def _create_model(self):
        """
        Create the classification model.
        """
        if self.model_type == 'logistic':
            base_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        elif self.model_type == 'naive_bayes':
            base_model = MultinomialNB()
        elif self.model_type == 'random_forest':
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic', 'naive_bayes', or 'random_forest'")
        
        self.model = MultiOutputClassifier(base_model)
    
    def prepare_data(self, df: pd.DataFrame, text_column: str, genre_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column
            genre_columns (List[str]): List of genre column names
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        X = df[text_column].values
        y = df[genre_columns].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, text_column: str, genre_columns: List[str], 
              test_size=0.2, random_state=42, min_samples_per_class=2):
        """
        Train the movie genre prediction model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column
            genre_columns (List[str]): List of genre column names
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            min_samples_per_class (int): Minimum samples required per class
        """
        print(f"Training {self.model_type} model with {self.vectorizer_type} vectorizer...")
        
        # Filter genres with sufficient data
        valid_genres = []
        for genre in genre_columns:
            positive_samples = df[genre].sum()
            negative_samples = len(df) - positive_samples
            if positive_samples >= min_samples_per_class and negative_samples >= min_samples_per_class:
                valid_genres.append(genre)
            else:
                print(f"⚠️  Skipping {genre}: insufficient data (pos: {positive_samples}, neg: {negative_samples})")
        
        if not valid_genres:
            raise ValueError("No genres have sufficient data for training!")
        
        print(f"✅ Using {len(valid_genres)} genres with sufficient data")
        
        # Create vectorizer and model
        self._create_vectorizer()
        self._create_model()
        
        # Prepare data with valid genres only
        X, y = self.prepare_data(df, text_column, valid_genres)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit vectorizer on training data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Calculate metrics
        self.genre_columns = valid_genres
        self.is_trained = True
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        print("Training completed!")
        
        return self.evaluate_model()
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        metrics['precision_macro'] = precision_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        
        # Micro averages
        metrics['precision_micro'] = precision_score(self.y_test, self.y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(self.y_test, self.y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(self.y_test, self.y_pred, average='micro', zero_division=0)
        
        # Per-genre metrics
        genre_metrics = {}
        for i, genre in enumerate(self.genre_columns):
            genre_metrics[genre] = {
                'precision': precision_score(self.y_test[:, i], self.y_pred[:, i], zero_division=0),
                'recall': recall_score(self.y_test[:, i], self.y_pred[:, i], zero_division=0),
                'f1': f1_score(self.y_test[:, i], self.y_pred[:, i], zero_division=0)
            }
        
        metrics['genre_metrics'] = genre_metrics
        
        return metrics
    
    def predict(self, text: str, threshold=0.5) -> List[str]:
        """
        Predict genres for a given text.
        
        Args:
            text (str): Input text (movie plot)
            threshold (float): Probability threshold for genre prediction
            
        Returns:
            List[str]: Predicted genres
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([text])
        
        # Get predictions
        predictions = self.model.predict(text_vectorized)
        probabilities = self.model.predict_proba(text_vectorized)
        
        # Get predicted genres
        predicted_genres = []
        for i, pred in enumerate(predictions[0]):
            if pred == 1:
                # Get the genre name (remove 'genre_' prefix)
                genre_name = self.genre_columns[i].replace('genre_', '').replace('_', ' ').title()
                predicted_genres.append(genre_name)
        
        return predicted_genres
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get probability predictions for all genres.
        
        Args:
            text (str): Input text (movie plot)
            
        Returns:
            Dict[str, float]: Dictionary with genre probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([text])
        
        # Get probabilities
        probabilities = self.model.predict_proba(text_vectorized)
        
        # Create dictionary with genre probabilities
        genre_probs = {}
        for i, prob in enumerate(probabilities):
            genre_name = self.genre_columns[i].replace('genre_', '').replace('_', ' ').title()
            genre_probs[genre_name] = float(prob[0][1])  # Probability of positive class
        
        return genre_probs
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'genre_columns': self.genre_columns,
            'vectorizer_type': self.vectorizer_type,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.genre_columns = model_data['genre_columns']
        self.vectorizer_type = model_data['vectorizer_type']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

def compare_models(df: pd.DataFrame, text_column: str, genre_columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compare different model configurations.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        genre_columns (List[str]): List of genre column names
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary with model comparison results
    """
    results = {}
    
    # Model configurations to test
    configs = [
        ('tfidf', 'logistic'),
        ('tfidf', 'naive_bayes'),
        ('tfidf', 'random_forest'),
        ('count', 'logistic'),
        ('count', 'naive_bayes'),
        ('count', 'random_forest')
    ]
    
    for vectorizer_type, model_type in configs:
        print(f"\nTesting {model_type} with {vectorizer_type} vectorizer...")
        
        predictor = MovieGenrePredictor(vectorizer_type=vectorizer_type, model_type=model_type)
        metrics = predictor.train(df, text_column, genre_columns)
        
        config_name = f"{vectorizer_type}_{model_type}"
        results[config_name] = {
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro']
        }
    
    return results

def get_feature_importance(model: MovieGenrePredictor, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """
    Get feature importance for the trained model.
    
    Args:
        model (MovieGenrePredictor): Trained model
        top_n (int): Number of top features to return per genre
        
    Returns:
        Dict[str, List[Tuple[str, float]]]: Dictionary with feature importance per genre
    """
    if not model.is_trained:
        raise ValueError("Model must be trained before getting feature importance")
    
    if model.model_type != 'logistic':
        print("Feature importance is only available for logistic regression models")
        return {}
    
    feature_names = model.vectorizer.get_feature_names_out()
    importance_dict = {}
    
    for i, genre in enumerate(model.genre_columns):
        if hasattr(model.model.estimators_[i], 'coef_'):
            coefficients = model.model.estimators_[i].coef_[0]
            feature_importance = list(zip(feature_names, coefficients))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            importance_dict[genre] = feature_importance[:top_n]
    
    return importance_dict

def print_evaluation_report(metrics: Dict[str, Any]):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics (Dict[str, Any]): Evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Micro): {metrics['precision_micro']:.4f}")
    print(f"Recall (Micro): {metrics['recall_micro']:.4f}")
    print(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
    
    print(f"\nPer-Genre Metrics:")
    print("-" * 50)
    for genre, genre_metrics in metrics['genre_metrics'].items():
        genre_name = genre.replace('genre_', '').replace('_', ' ').title()
        print(f"{genre_name:20} | Precision: {genre_metrics['precision']:.4f} | "
              f"Recall: {genre_metrics['recall']:.4f} | F1: {genre_metrics['f1']:.4f}") 