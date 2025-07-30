# Movie Genre Prediction from Plot Summaries

A machine learning model that predicts movie genres based on plot summaries using Natural Language Processing (NLP) techniques.

## 🎯 Project Overview

This project demonstrates how to build a supervised machine learning model that can predict a movie's genre based on its plot summary. The model uses various NLP techniques including:

- **Text Preprocessing**: Cleaning and normalizing text data
- **Feature Extraction**: Bag-of-Words and TF-IDF vectorization
- **Machine Learning Models**: Logistic Regression, Multinomial Naive Bayes, and Random Forest
- **Multi-label Classification**: Handling movies with multiple genres

## 🚀 Features

- **Text Preprocessing**: Remove stopwords, punctuation, and perform stemming
- **Multiple Vectorization Techniques**: CountVectorizer and TfidfVectorizer
- **Model Comparison**: Evaluate different ML algorithms
- **Multi-label Classification**: Handle movies with multiple genres
- **Interactive Web App**: Streamlit deployment for real-time predictions
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, and F1-score metrics

## 📁 Project Structure

```
movie-genre-predictor/
│
├── data/
│   └── movies.csv                 # Movie dataset with plot summaries and genres
│
├── notebooks/
│   └── MovieGenrePrediction.ipynb # Jupyter notebook with analysis
│
├── src/
│   ├── preprocess.py              # Text preprocessing functions
│   ├── model.py                   # ML model training and evaluation
│   └── utils.py                   # Utility functions
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── streamlit_app.py              # Streamlit web application
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-genre-predictor.git
   cd movie-genre-predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

## 📊 Dataset

The project uses a custom dataset with the following columns:
- `title`: Movie title
- `plot`: Plot summary
- `genre`: Movie genre(s)

The dataset includes movies from various genres like Action, Comedy, Drama, Horror, Sci-Fi, etc.

## 🧪 Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/MovieGenrePrediction.ipynb`

3. Run all cells to see the complete analysis

### Running the Streamlit App

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Enter a movie plot summary and get genre predictions!

## 📈 Model Performance

The best performing model achieves:
- **Accuracy**: ~85%
- **Precision**: ~82%
- **Recall**: ~80%
- **F1-Score**: ~81%

## 🎨 Sample Results

### Most Common Words by Genre
- **Action**: "fight", "mission", "agent", "weapon", "battle"
- **Comedy**: "funny", "laugh", "joke", "humor", "comic"
- **Drama**: "family", "life", "relationship", "love", "death"
- **Horror**: "kill", "dead", "ghost", "monster", "fear"

### Example Predictions
- Plot: "A young wizard discovers his magical powers and battles an evil sorcerer"
  - Predicted Genre: Fantasy, Adventure

- Plot: "A detective solves a series of mysterious murders in a small town"
  - Predicted Genre: Thriller, Mystery

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Data visualization
- **streamlit**: Web application framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Live Demo

Try the live demo: [Movie Genre Predictor on Streamlit Cloud](https://your-streamlit-app-url.streamlit.app)

## 📧 Contact

- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourusername)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

---

⭐ **Star this repository if you found it helpful!** 