#!/usr/bin/env python3
"""
Movie Genre Prediction Model Training Script

This script trains the movie genre prediction model and saves it for later use.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.preprocess import TextPreprocessor, create_genre_columns
from src.model import MovieGenrePredictor, print_evaluation_report

def main():
    """Main training function."""
    
    print("🎬 Movie Genre Predictor - Model Training")
    print("=" * 50)
    
    # Check if data exists
    data_path = Path('data/movies.csv')
    if not data_path.exists():
        print("❌ Error: data/movies.csv not found!")
        print("Please make sure the dataset exists in the data/ directory.")
        return
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} movies")
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=False)
    df_processed = preprocessor.preprocess_dataframe(df, 'plot')
    df_processed = create_genre_columns(df_processed, 'genre')
    df_processed = df_processed.dropna(subset=['plot_processed'])
    
    # Get genre columns
    genre_columns = [col for col in df_processed.columns if col.startswith('genre_')]
    print(f"✅ Preprocessed data with {len(genre_columns)} genres")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train model
    print("🤖 Training model...")
    predictor = MovieGenrePredictor(vectorizer_type='tfidf', model_type='logistic')
    metrics = predictor.train(df_processed, 'plot_processed', genre_columns, min_samples_per_class=3)
    
    # Print evaluation report
    print_evaluation_report(metrics)
    
    # Save model
    model_path = 'models/movie_genre_predictor.pkl'
    predictor.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_plots = [
        "A young wizard discovers his magical powers and battles an evil sorcerer",
        "A detective solves a series of mysterious murders in a small town",
        "A group of friends go on a hilarious road trip across the country"
    ]
    
    for i, plot in enumerate(test_plots, 1):
        predicted_genres = predictor.predict(plot)
        print(f"Test {i}: {plot}")
        print(f"Predicted: {', '.join(predicted_genres)}")
        print()
    
    print("🎉 Training completed successfully!")
    print("\nTo run the Streamlit app:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 