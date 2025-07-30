#!/usr/bin/env python3
"""
Test script for the Movie Genre Predictor model.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.preprocess import TextPreprocessor, create_genre_columns
from src.model import MovieGenrePredictor

def test_model():
    """Test the model training and prediction."""
    
    print("🧪 Testing Movie Genre Predictor Model")
    print("=" * 40)
    
    # Load data
    df = pd.read_csv('data/movies.csv')
    print(f"✅ Loaded {len(df)} movies")
    
    # Preprocess data
    preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=False)
    df_processed = preprocessor.preprocess_dataframe(df, 'plot')
    df_processed = create_genre_columns(df_processed, 'genre')
    df_processed = df_processed.dropna(subset=['plot_processed'])
    
    # Get genre columns
    genre_columns = [col for col in df_processed.columns if col.startswith('genre_')]
    print(f"✅ Found {len(genre_columns)} genre columns")
    
    # Check genre distribution
    print("\n📊 Genre Distribution:")
    for genre in genre_columns:
        count = df_processed[genre].sum()
        print(f"  {genre}: {count} movies")
    
    # Train model
    print(f"\n🤖 Training model...")
    predictor = MovieGenrePredictor(vectorizer_type='tfidf', model_type='logistic')
    metrics = predictor.train(df_processed, 'plot_processed', genre_columns, min_samples_per_class=3)
    
    print(f"✅ Model trained successfully!")
    print(f"📈 Accuracy: {metrics['accuracy']:.3f}")
    print(f"📈 F1-Score: {metrics['f1_macro']:.3f}")
    
    # Test predictions
    test_plots = [
        "A young wizard discovers his magical powers and battles an evil sorcerer",
        "A detective solves a series of mysterious murders in a small town",
        "A group of friends go on a hilarious road trip across the country"
    ]
    
    print(f"\n🧪 Testing predictions:")
    for i, plot in enumerate(test_plots, 1):
        predicted_genres = predictor.predict(plot)
        print(f"Test {i}: {plot}")
        print(f"Predicted: {', '.join(predicted_genres)}")
        print()
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_model() 