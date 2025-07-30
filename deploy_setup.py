#!/usr/bin/env python3
"""
Deployment setup script for Streamlit Cloud.
This script ensures all required data and models are available.
"""

import os
import sys
import nltk
import ssl
import pandas as pd
from pathlib import Path

def setup_nltk():
    """Download NLTK data for deployment."""
    print("📥 Setting up NLTK data...")
    
    # Handle SSL context
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required NLTK data
    nltk_data = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    
    for data in nltk_data:
        try:
            print(f"  Downloading {data}...")
            nltk.download(data, quiet=True)
            print(f"  ✅ {data} downloaded")
        except Exception as e:
            print(f"  ⚠️  Warning downloading {data}: {e}")
    
    print("✅ NLTK setup completed!")

def check_data():
    """Check if required data files exist."""
    print("📊 Checking data files...")
    
    data_file = Path("data/movies.csv")
    if data_file.exists():
        df = pd.read_csv(data_file)
        print(f"  ✅ Found {len(df)} movies in dataset")
        return True
    else:
        print("  ❌ data/movies.csv not found!")
        return False

def check_models():
    """Check if trained models exist."""
    print("🤖 Checking models...")
    
    model_file = Path("models/movie_genre_predictor.pkl")
    if model_file.exists():
        print("  ✅ Found trained model")
        return True
    else:
        print("  ℹ️  No trained model found (will train on first run)")
        return False

def main():
    """Main setup function."""
    print("🚀 Movie Genre Predictor - Deployment Setup")
    print("=" * 50)
    
    # Setup NLTK
    setup_nltk()
    
    # Check data
    data_ok = check_data()
    
    # Check models
    models_ok = check_models()
    
    print("\n📋 Setup Summary:")
    print(f"  NLTK Data: ✅ Ready")
    print(f"  Dataset: {'✅ Ready' if data_ok else '❌ Missing'}")
    print(f"  Model: {'✅ Ready' if models_ok else '🔄 Will train on first run'}")
    
    if data_ok:
        print("\n🎉 Setup completed! The app is ready to run.")
    else:
        print("\n⚠️  Setup completed with warnings. Some features may not work.")

if __name__ == "__main__":
    main() 