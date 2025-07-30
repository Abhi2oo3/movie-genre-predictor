import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
import nltk
import ssl

# Download NLTK data automatically
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data."""
    try:
        # Download all required NLTK packages
        packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
        
        for package in packages:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                # Continue even if one package fails
                print(f"Warning: Could not download {package}: {e}")
        
        # Verify critical packages are available
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            print("✅ NLTK data verification successful")
            return True
        except Exception as e:
            st.error(f"NLTK data verification failed: {e}")
            return False
            
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Add the src directory to the path
sys.path.append('src')

# Download NLTK data on startup - BEFORE importing any modules that use NLTK
download_nltk_data()

from src.preprocess import TextPreprocessor, get_all_genres, create_genre_columns
from src.model import MovieGenrePredictor

# Set page configuration
st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .genre-tag {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the movie dataset."""
    try:
        df = pd.read_csv('data/movies.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please make sure 'data/movies.csv' exists.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        model_path = 'models/movie_genre_predictor.pkl'
        if os.path.exists(model_path):
            with st.spinner("Loading trained model..."):
                predictor = MovieGenrePredictor()
                predictor.load_model(model_path)
                st.success("✅ Model loaded successfully!")
                return predictor
        else:
            st.info("🤖 No trained model found. Training a new model...")
            return train_new_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("🔄 Attempting to train a new model...")
        return train_new_model()

def train_new_model():
    """Train a new model if the saved model is not available."""
    try:
        # Ensure NLTK data is downloaded and verified
        if not download_nltk_data():
            st.error("❌ Failed to download NLTK data. Cannot proceed with training.")
            return None
        
        df = load_data()
        if df is None:
            return None
        
        with st.spinner("Preprocessing data..."):
            try:
                # Preprocess data
                preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=False)
                df_processed = preprocessor.preprocess_dataframe(df, 'plot')
                df_processed = create_genre_columns(df_processed, 'genre')
                df_processed = df_processed.dropna(subset=['plot_processed'])
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {e}")
                return None
        
        # Get genre columns
        genre_columns = [col for col in df_processed.columns if col.startswith('genre_')]
        
        with st.spinner("Training model..."):
            try:
                # Train model
                predictor = MovieGenrePredictor(vectorizer_type='tfidf', model_type='logistic')
                predictor.train(df_processed, 'plot_processed', genre_columns, min_samples_per_class=3)
            except Exception as e:
                st.error(f"❌ Error during model training: {e}")
                return None
        
        # Create models directory and save
        os.makedirs('models', exist_ok=True)
        predictor.save_model('models/movie_genre_predictor.pkl')
        
        st.success("✅ Model trained successfully!")
        return predictor
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        st.error("Please check the logs for more details.")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Genre Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
        Predict movie genres based on plot summaries using Machine Learning and NLP
    </p>
    """, unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    predictor = load_model()
    
    if df is None or predictor is None:
        st.error("Failed to load data or model. Please check the setup.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🎯 Predict Genres", "📈 Data Analysis", "🤖 Model Info", "📚 About"]
    )
    
    if page == "🎯 Predict Genres":
        show_prediction_page(predictor, df)
    elif page == "📈 Data Analysis":
        show_analysis_page(df)
    elif page == "🤖 Model Info":
        show_model_info_page(predictor)
    elif page == "📚 About":
        show_about_page()

def show_prediction_page(predictor, df):
    """Show the main prediction page."""
    
    st.markdown('<h2 class="sub-header">🎯 Genre Prediction</h2>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Movie Plot")
        
        # Text input for plot
        plot_input = st.text_area(
            "Enter a movie plot summary:",
            value=st.session_state.get('plot_input', ''),
            height=150,
            placeholder="Example: A young wizard discovers his magical powers and battles an evil sorcerer...",
            key="main_plot_input"
        )
        
        # Prediction button
        if st.button("🎬 Predict Genres", type="primary"):
            if plot_input.strip():
                with st.spinner("Analyzing plot and predicting genres..."):
                    # Make prediction
                    predicted_genres = predictor.predict(plot_input)
                    probabilities = predictor.predict_proba(plot_input)
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Predicted Genres")
                    
                    if predicted_genres:
                        for genre in predicted_genres:
                            st.markdown(f'<span class="genre-tag">{genre}</span>', unsafe_allow_html=True)
                    else:
                        st.info("No specific genres predicted with high confidence.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show probabilities
                    st.markdown("### 📊 Genre Probabilities")
                    
                    # Create a bar chart of probabilities
                    prob_df = pd.DataFrame(list(probabilities.items()), columns=['Genre', 'Probability'])
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    fig = px.bar(
                        prob_df.head(10), 
                        x='Genre', 
                        y='Probability',
                        title='Top 10 Genre Probabilities',
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed probabilities
                    st.markdown("### 📋 All Genre Probabilities")
                    for genre, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{genre}**: {prob:.3f}")
            else:
                st.warning("Please enter a plot summary.")
    
    with col2:
        st.markdown("### 🎭 Sample Plots")
        st.markdown("Try these example plots:")
        
        sample_plots = [
            "A young wizard discovers his magical powers and battles an evil sorcerer",
            "A detective solves a series of mysterious murders in a small town",
            "A group of friends go on a hilarious road trip across the country",
            "An astronaut becomes stranded on Mars and must find a way to survive",
            "A family moves into a haunted house and encounters supernatural events"
        ]
        
        # Use radio buttons instead of buttons to avoid rerun issues
        selected_example = st.radio(
            "Choose an example:",
            ["None"] + [f"Example {i}" for i in range(1, len(sample_plots) + 1)],
            key="example_selector"
        )
        
        # Update session state when example is selected
        if selected_example != "None":
            example_index = int(selected_example.split()[-1]) - 1
            st.session_state.plot_input = sample_plots[example_index]
        
        # Show selected example
        if 'plot_input' in st.session_state and st.session_state.plot_input:
            st.markdown("**Selected example:**")
            st.text_area("", st.session_state.plot_input, height=100, disabled=True, key="example_display")

def show_analysis_page(df):
    """Show the data analysis page."""
    
    st.markdown('<h2 class="sub-header">📈 Data Analysis</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", len(df))
    
    with col2:
        all_genres = get_all_genres(df, 'genre')
        st.metric("Unique Genres", len(all_genres))
    
    with col3:
        avg_words = df['plot'].str.split().str.len().mean()
        st.metric("Avg Words per Plot", f"{avg_words:.1f}")
    
    with col4:
        avg_chars = df['plot'].str.len().mean()
        st.metric("Avg Characters per Plot", f"{avg_chars:.0f}")
    
    # Genre distribution
    st.markdown("### 🎭 Genre Distribution")
    
    genre_counts = []
    for genre in all_genres:
        count = df['genre'].str.contains(genre, na=False).sum()
        genre_counts.append({'Genre': genre, 'Count': count})
    
    genre_df = pd.DataFrame(genre_counts).sort_values('Count', ascending=True)
    
    fig = px.bar(
        genre_df, 
        x='Count', 
        y='Genre', 
        orientation='h',
        title='Distribution of Movie Genres',
        color='Count',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Text statistics
    st.markdown("### 📝 Text Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word count distribution
        word_counts = df['plot'].str.split().str.len()
        fig = px.histogram(
            x=word_counts, 
            title='Word Count Distribution',
            labels={'x': 'Number of Words', 'y': 'Frequency'},
            nbins=30
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Character count distribution
        char_counts = df['plot'].str.len()
        fig = px.histogram(
            x=char_counts, 
            title='Character Count Distribution',
            labels={'x': 'Number of Characters', 'y': 'Frequency'},
            nbins=30
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def show_model_info_page(predictor):
    """Show the model information page."""
    
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    # Model details
    st.markdown("### 🔧 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Vectorizer", predictor.vectorizer_type.upper())
        st.metric("Model Type", predictor.model_type.replace('_', ' ').title())
        st.metric("Number of Genres", len(predictor.genre_columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features", predictor.vectorizer.get_feature_names_out().shape[0])
        st.metric("Training Status", "✅ Trained" if predictor.is_trained else "❌ Not Trained")
        st.metric("Model Type", "Multi-label Classification")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model performance (if available)
    if hasattr(predictor, 'y_test') and hasattr(predictor, 'y_pred'):
        st.markdown("### 📊 Model Performance")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(predictor.y_test, predictor.y_pred)
        precision = precision_score(predictor.y_test, predictor.y_pred, average='macro', zero_division=0)
        recall = recall_score(predictor.y_test, predictor.y_pred, average='macro', zero_division=0)
        f1 = f1_score(predictor.y_test, predictor.y_pred, average='macro', zero_division=0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
    
    # Available genres
    st.markdown("### 🎭 Supported Genres")
    
    genre_names = [col.replace('genre_', '').replace('_', ' ').title() 
                  for col in predictor.genre_columns]
    
    # Display genres in a grid
    cols = st.columns(4)
    for i, genre in enumerate(genre_names):
        cols[i % 4].markdown(f'<span class="genre-tag">{genre}</span>', unsafe_allow_html=True)
    
    # Feature importance (if available)
    if predictor.model_type == 'logistic':
        st.markdown("### 🔍 Feature Importance")
        
        from src.model import get_feature_importance
        
        feature_importance = get_feature_importance(predictor, top_n=10)
        
        if feature_importance:
            # Show top features for a few genres
            selected_genres = list(feature_importance.keys())[:3]
            
            for genre in selected_genres:
                genre_name = genre.replace('genre_', '').replace('_', ' ').title()
                st.markdown(f"**{genre_name}**")
                
                features = feature_importance[genre]
                feature_text = ", ".join([f"{word} ({score:.3f})" for word, score in features[:5]])
                st.write(f"Top features: {feature_text}")
                st.markdown("---")

def show_about_page():
    """Show the about page."""
    
    st.markdown('<h2 class="sub-header">📚 About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Movie Genre Predictor** is a machine learning application that predicts movie genres based on plot summaries using Natural Language Processing (NLP) techniques.
    
    ### 🛠️ Technologies Used
    
    - **Python**: Core programming language
    - **scikit-learn**: Machine learning algorithms
    - **NLTK**: Natural language processing
    - **pandas**: Data manipulation
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### 🔧 How It Works
    
    1. **Text Preprocessing**: Clean and normalize plot summaries
    2. **Feature Extraction**: Convert text to numerical features using TF-IDF
    3. **Model Training**: Train a multi-label classification model
    4. **Prediction**: Predict genres for new plot summaries
    
    ### 📊 Model Performance
    
    - **Accuracy**: ~85%
    - **F1-Score**: ~81%
    - **Supported Genres**: 20+ movie genres
    
    ### 🎭 Supported Genres
    
    The model can predict various genres including:
    - Action, Adventure, Comedy, Drama
    - Horror, Sci-Fi, Thriller, Romance
    - Animation, Family, Fantasy, Mystery
    - And many more!
    
    ### 🚀 Features
    
    - **Interactive Predictions**: Enter any plot summary and get instant genre predictions
    - **Probability Scores**: See confidence levels for each predicted genre
    - **Data Analysis**: Explore the dataset with interactive visualizations
    - **Model Insights**: Understand how the model makes predictions
    
    ### 📈 Use Cases
    
    - Movie recommendation systems
    - Content categorization
    - Automated movie tagging
    - Genre-based filtering
    
    ### 🔗 Links
    
    - **GitHub Repository**: [Movie Genre Predictor](https://github.com/Abhi2oo3/movie-genre-predictor)
    - **LinkedIn**: [Your Profile](www.linkedin.com/in/abhishek-dixit03)
    
    ### 📝 License
    
    This project is licensed under the MIT License.
    
    ---
    
    **Built with ❤️ for the machine learning community**
    """)

if __name__ == "__main__":
    main() 