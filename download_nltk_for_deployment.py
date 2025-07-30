#!/usr/bin/env python3
"""
Script to download NLTK data for immediate deployment use.
This ensures NLTK data is available without waiting for downloads.
"""

import nltk
import ssl
import os
import shutil
from pathlib import Path

def download_nltk_data():
    """Download all required NLTK data."""
    
    # Handle SSL context
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("📥 Downloading NLTK data for deployment...")
    
    # List of all required NLTK packages
    nltk_packages = [
        'stopwords',
        'punkt', 
        'wordnet',
        'omw-1.4',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    success_count = 0
    for package in nltk_packages:
        try:
            print(f"  Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"  ✅ {package} downloaded successfully")
            success_count += 1
        except Exception as e:
            print(f"  ⚠️  Warning downloading {package}: {e}")
    
    print(f"\n📊 Download Summary:")
    print(f"  Successfully downloaded: {success_count}/{len(nltk_packages)} packages")
    
    # Verify critical packages
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        print("  ✅ Critical packages verified")
        return True
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False

def create_nltk_data_dir():
    """Create a local NLTK data directory for deployment."""
    
    # Create nltk_data directory in project
    nltk_data_dir = Path("nltk_data")
    nltk_data_dir.mkdir(exist_ok=True)
    
    # Set NLTK data path to local directory
    nltk.data.path.append(str(nltk_data_dir.absolute()))
    
    print(f"📁 Created local NLTK data directory: {nltk_data_dir.absolute()}")
    return nltk_data_dir

def main():
    """Main function."""
    print("🚀 NLTK Data Download for Deployment")
    print("=" * 40)
    
    # Create local NLTK data directory
    nltk_data_dir = create_nltk_data_dir()
    
    # Download all required data
    success = download_nltk_data()
    
    if success:
        print("\n🎉 NLTK data download completed successfully!")
        print("The app should now work without NLTK download delays.")
    else:
        print("\n⚠️  NLTK data download completed with warnings.")
        print("Some features may not work properly.")

if __name__ == "__main__":
    main() 