import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(data_path, sample_frac=0.1, use_chunks=False, chunk_size=1000000):
    """
    Loads ratings and movies data with optimized numpy operations.
    
    Args:
        data_path (str): Path to the ml-32m directory.
        sample_frac (float): Fraction of data to load (for testing/speed).
        use_chunks (bool): Whether to use chunked reading for large datasets.
        chunk_size (int): Size of chunks when use_chunks=True.
        
    Returns:
        dict: Dictionary containing train/test split and mappings with numpy arrays.
    """
    ratings_path = os.path.join(data_path, "ratings.csv")
    print(f"Loading ratings from {ratings_path}...")
    
    if use_chunks and sample_frac >= 0.5:
        # For large datasets, use chunked reading to save memory
        print(f"Using chunked reading with chunk_size={chunk_size:,}...")
        chunks = []
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            chunks.append(chunk)
        ratings_df = pd.concat(chunks, ignore_index=True)
    else:
        # Standard loading
        ratings_df = pd.read_csv(ratings_path)
        if sample_frac < 1.0:
            ratings_df = ratings_df.sample(frac=sample_frac, random_state=42)
        
    print(f"Loaded {len(ratings_df):,} ratings.")

    # Encode User IDs - Convert to numpy array directly
    print("Encoding User IDs...")
    user_encoder = LabelEncoder()
    user_indices = user_encoder.fit_transform(ratings_df['userId'].values)
    n_users = len(user_encoder.classes_)
    
    # Encode Movie IDs - Convert to numpy array directly
    print("Encoding Movie IDs...")
    movie_encoder = LabelEncoder()
    movie_indices = movie_encoder.fit_transform(ratings_df['movieId'].values)
    n_movies = len(movie_encoder.classes_)
    
    print(f"Num users: {n_users:,}, Num movies: {n_movies:,}")
    
    # Prepare features and targets as numpy arrays (more efficient)
    # Stack arrays using numpy for better performance
    X = np.column_stack([user_indices, movie_indices]).astype(np.int32)
    y = ratings_df['rating'].values.astype(np.float32)
    
    # Get rating statistics using numpy
    min_rating = np.min(y)
    max_rating = np.max(y)
    
    # Split data using numpy indexing for better performance
    print("Splitting data...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"Training samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'n_users': n_users,
        'n_movies': n_movies,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'min_rating': float(min_rating),
        'max_rating': float(max_rating)
    }

def load_movies(data_path):
    """Load movies data and return as DataFrame."""
    movies_path = os.path.join(data_path, "movies.csv")
    return pd.read_csv(movies_path)
