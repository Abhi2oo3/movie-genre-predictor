import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import wordcloud, but make it optional
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

def plot_genre_distribution(df: pd.DataFrame, genre_column: str, figsize=(12, 8)):
    """
    Plot the distribution of genres in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        genre_column (str): Name of the genre column
        figsize (tuple): Figure size
    """
    from src.preprocess import extract_genres, get_all_genres
    
    # Get all genres
    all_genres = get_all_genres(df, genre_column)
    
    # Count occurrences of each genre
    genre_counts = []
    for genre in all_genres:
        count = df[genre_column].str.contains(genre, na=False).sum()
        genre_counts.append(count)
    
    # Create DataFrame for plotting
    genre_df = pd.DataFrame({
        'Genre': all_genres,
        'Count': genre_counts
    }).sort_values('Count', ascending=True)
    
    # Create plot
    plt.figure(figsize=figsize)
    bars = plt.barh(genre_df['Genre'], genre_df['Count'])
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center')
    
    plt.title('Distribution of Movie Genres', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Movies', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return genre_df

def plot_text_statistics(df: pd.DataFrame, text_column: str, figsize=(15, 5)):
    """
    Plot text statistics (word count, character count distributions).
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        figsize (tuple): Figure size
    """
    # Calculate statistics
    word_counts = df[text_column].str.split().str.len()
    char_counts = df[text_column].str.len()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Word count distribution
    axes[0].hist(word_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(word_counts.mean(), color='red', linestyle='--', 
                   label=f'Mean: {word_counts.mean():.1f}')
    axes[0].set_title('Word Count Distribution')
    axes[0].set_xlabel('Number of Words')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Character count distribution
    axes[1].hist(char_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].axvline(char_counts.mean(), color='red', linestyle='--', 
                   label=f'Mean: {char_counts.mean():.1f}')
    axes[1].set_title('Character Count Distribution')
    axes[1].set_xlabel('Number of Characters')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Box plot of word counts by genre
    from src.preprocess import get_all_genres
    all_genres = get_all_genres(df, 'genre')[:10]  # Top 10 genres
    
    genre_word_counts = []
    genre_labels = []
    
    for genre in all_genres:
        genre_movies = df[df['genre'].str.contains(genre, na=False)]
        if len(genre_movies) > 0:
            genre_word_counts.append(genre_movies[text_column].str.split().str.len())
            genre_labels.append(genre)
    
    if genre_word_counts:
        axes[2].boxplot(genre_word_counts, labels=genre_labels)
        axes[2].set_title('Word Count by Genre (Top 10)')
        axes[2].set_xlabel('Genre')
        axes[2].set_ylabel('Number of Words')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_wordcloud_by_genre(df: pd.DataFrame, text_column: str, genre_column: str, 
                            top_genres=6, figsize=(15, 10)):
    """
    Create word clouds for different genres.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        genre_column (str): Name of the genre column
        top_genres (int): Number of top genres to visualize
        figsize (tuple): Figure size
    """
    if not WORDCLOUD_AVAILABLE:
        print("⚠️  WordCloud not available. Install with: pip install wordcloud")
        return
    
    from src.preprocess import get_all_genres
    
    # Get top genres by count
    all_genres = get_all_genres(df, genre_column)
    genre_counts = []
    for genre in all_genres:
        count = df[genre_column].str.contains(genre, na=False).sum()
        genre_counts.append((genre, count))
    
    genre_counts.sort(key=lambda x: x[1], reverse=True)
    top_genres_list = [genre for genre, count in genre_counts[:top_genres]]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, genre in enumerate(top_genres_list):
        # Get all plots for this genre
        genre_plots = df[df[genre_column].str.contains(genre, na=False)][text_column]
        
        # Combine all text
        combined_text = ' '.join(genre_plots.dropna())
        
        if combined_text.strip():
            # Create word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(combined_text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{genre} Movies', fontsize=14, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No data for {genre}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{genre} Movies', fontsize=14, fontweight='bold')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(top_genres_list), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results: dict, figsize=(12, 8)):
    """
    Plot model comparison results.
    
    Args:
        results (dict): Model comparison results
        figsize (tuple): Figure size
    """
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Create DataFrame
    plot_data = []
    for model in models:
        for metric in metrics:
            plot_data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Score': results[model][metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric')
    plt.title('Model Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         genre_columns: list, figsize=(15, 12)):
    """
    Plot confusion matrices for each genre.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        genre_columns (list): List of genre column names
        figsize (tuple): Figure size
    """
    n_genres = len(genre_columns)
    n_cols = 3
    n_rows = (n_genres + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, genre in enumerate(genre_columns):
        genre_name = genre.replace('genre_', '').replace('_', ' ').title()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not ' + genre_name, genre_name],
                   yticklabels=['Not ' + genre_name, genre_name],
                   ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {genre_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_genres, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance: dict, top_n=15, figsize=(15, 10)):
    """
    Plot feature importance for each genre.
    
    Args:
        feature_importance (dict): Feature importance dictionary
        top_n (int): Number of top features to show
        figsize (tuple): Figure size
    """
    n_genres = len(feature_importance)
    n_cols = 2
    n_rows = (n_genres + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (genre, features) in enumerate(feature_importance.items()):
        genre_name = genre.replace('genre_', '').replace('_', ' ').title()
        
        # Get top features
        top_features = features[:top_n]
        words = [feature[0] for feature in top_features]
        scores = [feature[1] for feature in top_features]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(words))
        colors = ['red' if score < 0 else 'blue' for score in scores]
        
        axes[i].barh(y_pos, scores, color=colors, alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(words)
        axes[i].set_xlabel('Coefficient')
        axes[i].set_title(f'Feature Importance - {genre_name}')
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[i].grid(axis='x', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_genres, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_interactive_genre_distribution(df: pd.DataFrame, genre_column: str):
    """
    Create an interactive plot of genre distribution using Plotly.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        genre_column (str): Name of the genre column
    """
    from src.preprocess import get_all_genres
    
    # Get all genres
    all_genres = get_all_genres(df, genre_column)
    
    # Count occurrences of each genre
    genre_counts = []
    for genre in all_genres:
        count = df[genre_column].str.contains(genre, na=False).sum()
        genre_counts.append(count)
    
    # Create DataFrame for plotting
    genre_df = pd.DataFrame({
        'Genre': all_genres,
        'Count': genre_counts
    }).sort_values('Count', ascending=True)
    
    # Create interactive bar plot
    fig = px.bar(genre_df, x='Count', y='Genre', orientation='h',
                 title='Distribution of Movie Genres',
                 labels={'Count': 'Number of Movies', 'Genre': 'Genre'},
                 color='Count', color_continuous_scale='viridis')
    
    fig.update_layout(
        title_font_size=20,
        title_font_color='darkblue',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=600
    )
    
    fig.show()

def create_interactive_word_count_analysis(df: pd.DataFrame, text_column: str, genre_column: str):
    """
    Create an interactive analysis of word counts by genre.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        genre_column (str): Name of the genre column
    """
    from src.preprocess import get_all_genres
    
    # Get all genres
    all_genres = get_all_genres(df, genre_column)
    
    # Calculate word counts for each genre
    genre_data = []
    for genre in all_genres:
        genre_movies = df[df[genre_column].str.contains(genre, na=False)]
        if len(genre_movies) > 0:
            word_counts = genre_movies[text_column].str.split().str.len()
            for count in word_counts.dropna():
                genre_data.append({
                    'Genre': genre,
                    'Word Count': count
                })
    
    genre_df = pd.DataFrame(genre_data)
    
    # Create interactive box plot
    fig = px.box(genre_df, x='Genre', y='Word Count', 
                 title='Word Count Distribution by Genre',
                 color='Genre')
    
    fig.update_layout(
        title_font_size=20,
        title_font_color='darkblue',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    fig.show()

def save_plots_to_file(plot_function, filename: str, *args, **kwargs):
    """
    Save a plot to a file.
    
    Args:
        plot_function: Function that creates the plot
        filename (str): Name of the file to save
        *args: Arguments for the plot function
        **kwargs: Keyword arguments for the plot function
    """
    plt.figure()
    plot_function(*args, **kwargs)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")

def print_dataset_summary(df: pd.DataFrame, text_column: str, genre_column: str):
    """
    Print a comprehensive summary of the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column
        genre_column (str): Name of the genre column
    """
    from src.preprocess import get_all_genres, analyze_text_statistics
    
    print("="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of movies: {len(df)}")
    
    # Genre information
    all_genres = get_all_genres(df, genre_column)
    print(f"Number of unique genres: {len(all_genres)}")
    print(f"Genres: {', '.join(all_genres)}")
    
    # Text statistics
    stats = analyze_text_statistics(df, text_column)
    print(f"\nText Statistics:")
    print(f"Average words per plot: {stats['avg_words_per_plot']:.1f}")
    print(f"Min words per plot: {stats['min_words_per_plot']}")
    print(f"Max words per plot: {stats['max_words_per_plot']}")
    print(f"Average characters per plot: {stats['avg_chars_per_plot']:.1f}")
    
    # Missing values
    missing_text = df[text_column].isna().sum()
    missing_genre = df[genre_column].isna().sum()
    print(f"\nMissing Values:")
    print(f"Missing plot summaries: {missing_text}")
    print(f"Missing genres: {missing_genre}")
    
    # Genre distribution
    print(f"\nGenre Distribution:")
    for genre in all_genres:
        count = df[genre_column].str.contains(genre, na=False).sum()
        percentage = (count / len(df)) * 100
        print(f"{genre:20}: {count:3d} movies ({percentage:5.1f}%)")
    
    print("="*60) 