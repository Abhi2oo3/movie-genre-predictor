#!/usr/bin/env python3
"""
Test script to verify the app works locally before deployment.
"""

import subprocess
import sys
import os

def test_dependencies():
    """Test if all dependencies are available."""
    print("🧪 Testing dependencies...")
    
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import nltk
        import sklearn
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_nltk_data():
    """Test if NLTK data is available."""
    print("🧪 Testing NLTK data...")
    
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        # Test basic NLTK functionality
        text = "This is a test sentence."
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        print("✅ NLTK data working correctly")
        return True
    except Exception as e:
        print(f"❌ NLTK data issue: {e}")
        return False

def test_model_loading():
    """Test if model can be loaded."""
    print("🧪 Testing model loading...")
    
    try:
        import sys
        sys.path.append('src')
        from src.model import MovieGenrePredictor
        
        model_path = 'models/movie_genre_predictor.pkl'
        if os.path.exists(model_path):
            predictor = MovieGenrePredictor()
            predictor.load_model(model_path)
            print("✅ Model loaded successfully")
            return True
        else:
            print("⚠️  No trained model found (will train on first run)")
            return True
    except Exception as e:
        print(f"❌ Model loading issue: {e}")
        return False

def test_data_loading():
    """Test if data can be loaded."""
    print("🧪 Testing data loading...")
    
    try:
        import pandas as pd
        df = pd.read_csv('data/movies.csv')
        print(f"✅ Data loaded successfully ({len(df)} movies)")
        return True
    except Exception as e:
        print(f"❌ Data loading issue: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Movie Genre Predictor - Local Testing")
    print("=" * 40)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("NLTK Data", test_nltk_data),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  ❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        print("\n🚀 To run the app locally:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deployment.")
        return False
    
    return True

if __name__ == "__main__":
    main() 