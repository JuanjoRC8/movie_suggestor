import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data(data_path, sample_frac=0.1):
    """
    Loads ratings and movies data.
    
    Args:
        data_path (str): Path to the ml-32m directory.
        sample_frac (float): Fraction of data to load (for testing/speed).
        
    Returns:
        dict: Dictionary containing train/test split and mappings.
    """
    ratings_path = os.path.join(data_path, "ratings.csv")
    print(f"Loading ratings from {ratings_path}...")
    
    # Load a subset if sample_frac is small to save memory during dev
    # For full training, we might need a more robust pipeline (tf.data)
    # but pandas is fine for < 10M rows usually.
    
    if sample_frac < 1.0:
        # Read a random sample roughly by skipping rows? 
        # Actually, reading full csv then sampling is memory heavy.
        # Let's read full if memory allows, or use chunking.
        # For now, let's assume we can read it or just read first N rows for speed.
        # Better: read_csv with nrows for dev, or sample after load.
        # Given 32M rows, full load takes ~1-2GB RAM. Should be fine.
        ratings_df = pd.read_csv(ratings_path)
        ratings_df = ratings_df.sample(frac=sample_frac, random_state=42)
    else:
        ratings_df = pd.read_csv(ratings_path)
        
    print(f"Loaded {len(ratings_df)} ratings.")

    # Encode User IDs
    print("Encoding User IDs...")
    user_encoder = LabelEncoder()
    ratings_df['user'] = user_encoder.fit_transform(ratings_df['userId'])
    n_users = len(user_encoder.classes_)
    
    # Encode Movie IDs
    print("Encoding Movie IDs...")
    movie_encoder = LabelEncoder()
    ratings_df['movie'] = movie_encoder.fit_transform(ratings_df['movieId'])
    n_movies = len(movie_encoder.classes_)
    
    print(f"Num users: {n_users}, Num movies: {n_movies}")
    
    # Prepare features and targets
    X = ratings_df[['user', 'movie']].values
    y = ratings_df['rating'].values
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'n_users': n_users,
        'n_movies': n_movies,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'min_rating': ratings_df['rating'].min(),
        'max_rating': ratings_df['rating'].max()
    }

def load_movies(data_path):
    movies_path = os.path.join(data_path, "movies.csv")
    return pd.read_csv(movies_path)
