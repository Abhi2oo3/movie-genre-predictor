#!/usr/bin/env python3
"""
Quick Start Script for Movie Genre Predictor

This script helps you get started with the Movie Genre Predictor project.
It will install dependencies, download NLTK data, train the model, and run the app.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def download_nltk_data():
    """Download required NLTK data."""
    nltk_script = """
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
print("NLTK data downloaded successfully!")
"""
    
    with open("temp_nltk_download.py", "w") as f:
        f.write(nltk_script)
    
    success = run_command("python temp_nltk_download.py", "Downloading NLTK data")
    
    # Clean up
    if os.path.exists("temp_nltk_download.py"):
        os.remove("temp_nltk_download.py")
    
    return success

def train_model():
    """Train the movie genre prediction model."""
    return run_command("python train_model.py", "Training the model")

def run_streamlit():
    """Run the Streamlit app."""
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your browser automatically.")
    print("Press Ctrl+C to stop the app.")
    
    try:
        subprocess.run("streamlit run streamlit_app.py", shell=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

def main():
    """Main function to set up and run the project."""
    print("🎬 Movie Genre Predictor - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check your internet connection.")
        return
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Failed to download NLTK data. Please check your internet connection.")
        return
    
    # Train model
    if not train_model():
        print("❌ Failed to train the model. Please check the error messages above.")
        return
    
    # Ask user if they want to run the app
    print("\n🎉 Setup completed successfully!")
    response = input("Do you want to run the Streamlit app now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_streamlit()
    else:
        print("\n📝 To run the app later, use:")
        print("streamlit run streamlit_app.py")
    
    print("\n🎬 Happy coding!")

if __name__ == "__main__":
    main() 