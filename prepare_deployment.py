#!/usr/bin/env python3
"""
Comprehensive deployment preparation script.
This script will:
1. Download NLTK data
2. Train the model
3. Save everything for deployment
"""

import os
import sys
import subprocess
import nltk
import ssl
from pathlib import Path

def setup_nltk():
    """Download NLTK data."""
    print("📥 Setting up NLTK data...")
    
    # Handle SSL context
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download all required packages
    packages = [
        'stopwords', 'punkt', 'wordnet', 'omw-1.4', 
        'punkt_tab', 'averaged_perceptron_tagger'
    ]
    
    for package in packages:
        try:
            print(f"  Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"  ✅ {package} downloaded")
        except Exception as e:
            print(f"  ⚠️  Warning: {package} - {e}")
    
    print("✅ NLTK setup completed!")

def train_model():
    """Train the model."""
    print("🤖 Training model...")
    
    try:
        # Import after NLTK setup
        sys.path.append('src')
        from src.preprocess import TextPreprocessor, create_genre_columns
        from src.model import MovieGenrePredictor
        import pandas as pd
        
        # Load and preprocess data
        print("  Loading data...")
        df = pd.read_csv('data/movies.csv')
        
        print("  Preprocessing data...")
        preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=False)
        df_processed = preprocessor.preprocess_dataframe(df, 'plot')
        df_processed = create_genre_columns(df_processed, 'genre')
        df_processed = df_processed.dropna(subset=['plot_processed'])
        
        # Get genre columns
        genre_columns = [col for col in df_processed.columns if col.startswith('genre_')]
        print(f"  Found {len(genre_columns)} genres")
        
        # Train model
        print("  Training model...")
        predictor = MovieGenrePredictor(vectorizer_type='tfidf', model_type='logistic')
        metrics = predictor.train(df_processed, 'plot_processed', genre_columns, min_samples_per_class=3)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        predictor.save_model('models/movie_genre_predictor.pkl')
        
        print(f"  ✅ Model trained successfully!")
        print(f"  📊 Accuracy: {metrics['accuracy']:.3f}")
        print(f"  📊 F1-Score: {metrics['f1_macro']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error training model: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Movie Genre Predictor - Deployment Preparation")
    print("=" * 50)
    
    # Step 1: Setup NLTK
    setup_nltk()
    
    # Step 2: Train model
    if train_model():
        print("\n🎉 Deployment preparation completed successfully!")
        print("\n📋 Next steps:")
        print("1. Commit all changes: git add .")
        print("2. Commit: git commit -m 'Add trained model and NLTK data'")
        print("3. Push: git push origin main")
        print("4. Redeploy on Streamlit Cloud")
    else:
        print("\n❌ Deployment preparation failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 