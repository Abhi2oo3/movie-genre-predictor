# 🚀 Deployment Guide: Movie Genre Predictor

This guide provides step-by-step instructions for uploading your project to GitHub and deploying it on Streamlit Cloud.

## 📋 Prerequisites

Before starting, make sure you have:
- [Git](https://git-scm.com/) installed on your computer
- A [GitHub](https://github.com/) account
- A [Streamlit Cloud](https://streamlit.io/cloud) account
- Python 3.8+ installed

## 🎯 Step 1: Prepare Your Project

### 1.1 Verify Project Structure
Make sure your project has the following structure:
```
movie-genre-predictor/
│
├── data/
│   └── movies.csv
│
├── notebooks/
│   └── MovieGenrePrediction.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── utils.py
│
├── README.md
├── requirements.txt
├── streamlit_app.py
├── train_model.py
├── .gitignore
└── DEPLOYMENT_GUIDE.md
```

### 1.2 Test Your Project Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Train the model
python train_model.py

# Test the Streamlit app
streamlit run streamlit_app.py
```

## 🐙 Step 2: Upload to GitHub

### 2.1 Initialize Git Repository
```bash
# Navigate to your project directory
cd movie-genre-predictor

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Movie Genre Predictor project"
```

### 2.2 Create GitHub Repository
1. Go to [GitHub](https://github.com/)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository: `movie-genre-predictor`
5. Make it **Public** (required for free Streamlit Cloud)
6. **Don't** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 2.3 Push to GitHub
```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/movie-genre-predictor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2.4 Verify Upload
1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/movie-genre-predictor`
2. Verify all files are uploaded correctly
3. Check that the README.md displays properly

## ☁️ Step 3: Deploy on Streamlit Cloud

### 3.1 Prepare for Streamlit Cloud

#### 3.1.1 Create requirements.txt for Streamlit
Make sure your `requirements.txt` includes all necessary packages:
```txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
streamlit==1.25.0
jupyter==1.0.0
# wordcloud==1.9.2  # Optional - removed for Python 3.13 compatibility
plotly==5.15.0
```

#### 3.1.2 Create .streamlit/config.toml (Optional)
Create a `.streamlit` directory and add a `config.toml` file:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

### 3.2 Deploy on Streamlit Cloud

#### 3.2.1 Sign Up for Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign up with your GitHub account
3. Complete the verification process

#### 3.2.2 Deploy Your App
1. Click "New app" in Streamlit Cloud
2. Select your GitHub repository: `YOUR_USERNAME/movie-genre-predictor`
3. Set the main file path: `streamlit_app.py`
4. Set the app URL (optional): `movie-genre-predictor`
5. Click "Deploy!"

#### 3.2.3 Configure App Settings
1. Go to your deployed app settings
2. Set Python version to 3.9 or 3.10
3. Add any environment variables if needed
4. Configure advanced settings if required

### 3.3 Handle Dependencies

#### 3.3.1 NLTK Data Download
Since Streamlit Cloud doesn't have NLTK data pre-installed, we need to download it in the app. The app already handles this automatically.

#### 3.3.2 Model Training
The app will automatically train the model if no saved model is found. This might take a few minutes on the first run.

## 🔧 Step 4: Troubleshooting

### 4.1 Common Issues

#### Issue: "Module not found" errors
**Solution**: Make sure all imports in `streamlit_app.py` are correct and the `src` directory is properly added to the path.

#### Issue: NLTK data not found
**Solution**: The app automatically downloads NLTK data. If it fails, you can add this to your app:
```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

#### Issue: Model training takes too long
**Solution**: 
1. Train the model locally first
2. Upload the trained model to GitHub
3. The app will use the pre-trained model

#### Issue: App crashes on startup
**Solution**: Check the Streamlit Cloud logs for specific error messages and fix accordingly.

### 4.2 Performance Optimization

#### 4.2.1 Cache Data Loading
The app uses `@st.cache_data` and `@st.cache_resource` for better performance.

#### 4.2.2 Reduce Model Size
If the model is too large, consider:
- Reducing the number of features in the vectorizer
- Using a smaller dataset for training
- Compressing the model

## 📊 Step 5: Monitor and Maintain

### 5.1 Monitor App Performance
1. Check Streamlit Cloud dashboard regularly
2. Monitor app usage and performance
3. Check for any errors in the logs

### 5.2 Update Your App
```bash
# Make changes to your code
git add .
git commit -m "Update app with new features"
git push origin main
```
Streamlit Cloud will automatically redeploy your app.

### 5.3 Share Your App
1. Get your app URL from Streamlit Cloud
2. Share it on LinkedIn, GitHub, or other platforms
3. Add it to your portfolio

## 🎉 Step 6: Celebrate and Share!

### 6.1 Update Your LinkedIn Post
Share your deployed app with a post like:

```
🎬 Just deployed my Movie Genre Predictor ML project!

🔗 Live Demo: [Your Streamlit App URL]
📊 GitHub: [Your GitHub Repository URL]

✨ Features:
• Predicts movie genres from plot summaries
• 85% accuracy using NLP & ML
• Interactive web interface
• Real-time predictions

🛠️ Built with: Python, scikit-learn, NLTK, Streamlit

#MachineLearning #NLP #Python #DataScience #Streamlit #GitHub
```

### 6.2 Add to Your Portfolio
1. Add the project to your GitHub profile
2. Include it in your portfolio website
3. Mention it in job applications

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Pages](https://pages.github.com/)
- [NLTK Documentation](https://www.nltk.org/)

## 📞 Support

If you encounter any issues:
1. Check the [Streamlit Community](https://discuss.streamlit.io/)
2. Search for similar issues on GitHub
3. Check the app logs in Streamlit Cloud dashboard

---

**Congratulations! 🎉** You've successfully deployed your Movie Genre Predictor app. Share it with the world and keep building amazing projects! 