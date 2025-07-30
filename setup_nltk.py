#!/usr/bin/env python3
"""
Script to download NLTK data for deployment.
Run this locally to ensure NLTK data is available.
"""

import nltk
import ssl
import os

def download_nltk_data():
    """Download required NLTK data."""
    
    # Handle SSL context for downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("📥 Downloading NLTK data...")
    
    # List of required NLTK data
    nltk_data = [
        'stopwords',
        'punkt', 
        'wordnet',
        'omw-1.4'
    ]
    
    for data in nltk_data:
        try:
            print(f"  Downloading {data}...")
            nltk.download(data, quiet=True)
            print(f"  ✅ {data} downloaded successfully")
        except Exception as e:
            print(f"  ❌ Error downloading {data}: {e}")
    
    print("🎉 NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data() 