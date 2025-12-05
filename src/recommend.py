import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add src to path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import RecommenderNet
from src.data_loader import load_movies
import pickle
import argparse

def load_encoders():
    """Load the saved encoders."""
    with open('user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open('movie_encoder.pkl', 'rb') as f:
        movie_encoder = pickle.load(f)
    return user_encoder, movie_encoder

def get_recommendations(user_id, model, user_encoder, movie_encoder, movies_df, top_n=10, batch_size=10000):
    """
    Generate top N movie recommendations for a given user using vectorized operations.
    
    Args:
        user_id (int): Original user ID from the dataset.
        model: Trained RecommenderNet model.
        user_encoder: LabelEncoder for users.
        movie_encoder: LabelEncoder for movies.
        movies_df: DataFrame containing movie information.
        top_n (int): Number of recommendations to return.
        batch_size (int): Batch size for predictions (for memory efficiency).
        
    Returns:
        DataFrame with recommended movies.
    """
    # Encode the user ID
    try:
        user_encoded = user_encoder.transform(np.array([user_id]))[0]
    except ValueError:
        print(f"User ID {user_id} not found in training data.")
        return None
    
    # Get all movie IDs using numpy array
    all_movie_ids = movie_encoder.classes_
    n_movies = len(all_movie_ids)
    all_movie_encoded = np.arange(n_movies, dtype=np.int32)
    
    # Create input pairs (user, movie) for all movies using numpy broadcasting
    # More efficient than np.full
    user_array = np.repeat(user_encoded, n_movies).astype(np.int32)
    
    # Stack into input format using numpy (faster than column_stack for large arrays)
    inputs = np.stack([user_array, all_movie_encoded], axis=1)
    
    # Predict ratings in batches for memory efficiency
    print(f"Predicting ratings for {n_movies:,} movies...")
    predictions = np.empty(n_movies, dtype=np.float32)
    
    for i in range(0, n_movies, batch_size):
        end_idx = min(i + batch_size, n_movies)
        batch = inputs[i:end_idx]
        predictions[i:end_idx] = model.predict(batch, verbose=0).flatten()
    
    # Get top N movie indices using numpy's argpartition (faster than argsort for top-k)
    # argpartition is O(n) vs argsort O(n log n)
    if top_n < n_movies:
        top_indices = np.argpartition(predictions, -top_n)[-top_n:]
        # Sort only the top N
        top_indices = top_indices[np.argsort(predictions[top_indices])[::-1]]
    else:
        top_indices = np.argsort(predictions)[::-1][:top_n]
    
    # Get the original movie IDs and ratings using numpy indexing
    recommended_movie_ids = all_movie_ids[top_indices]
    predicted_ratings = predictions[top_indices]
    
    # Get movie details using vectorized pandas operations
    recommendations = movies_df[movies_df['movieId'].isin(recommended_movie_ids)].copy()
    
    # Create a mapping for efficient lookup
    rating_map = dict(zip(recommended_movie_ids, predicted_ratings))
    recommendations['predicted_rating'] = recommendations['movieId'].map(rating_map)
    
    # Sort by predicted rating
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    return recommendations[['movieId', 'title', 'genres', 'predicted_rating']]

def get_batch_recommendations(user_ids, model, user_encoder, movie_encoder, movies_df, top_n=10, n_workers=None):
    """
    Generate recommendations for multiple users in parallel.
    
    Args:
        user_ids (list): List of user IDs.
        model: Trained RecommenderNet model.
        user_encoder: LabelEncoder for users.
        movie_encoder: LabelEncoder for movies.
        movies_df: DataFrame containing movie information.
        top_n (int): Number of recommendations per user.
        n_workers (int): Number of parallel workers (default: CPU count).
        
    Returns:
        dict: Dictionary mapping user_id to recommendations DataFrame.
    """
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(user_ids))
    
    print(f"Generating recommendations for {len(user_ids)} users using {n_workers} workers...")
    
    def get_recs_for_user(uid):
        return uid, get_recommendations(uid, model, user_encoder, movie_encoder, movies_df, top_n)
    
    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for user_id, recs in executor.map(get_recs_for_user, user_ids):
            results[user_id] = recs
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate Movie Recommendations')
    parser.add_argument('--user_id', type=int, required=True, help='User ID to generate recommendations for')
    parser.add_argument('--top_n', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for predictions')
    
    args = parser.parse_args()
    
    # Load model configuration from numpy
    print("Loading model configuration...")
    config = np.load('model_config.npy')
    num_users = int(config[0])
    num_movies = int(config[1])
    embedding_size = int(config[2])
    
    # Load encoders
    print("Loading encoders...")
    user_encoder, movie_encoder = load_encoders()
    
    # Load movies
    print("Loading movies...")
    movies_df = load_movies("ml-32m")
    
    print(f"Initializing model ({num_users:,} users, {num_movies:,} movies, embedding_size={embedding_size})...")
    model = RecommenderNet(num_users, num_movies, embedding_size)
    
    # Build the model by calling it once
    dummy_input = np.array([[0, 0]], dtype=np.int32)
    _ = model(dummy_input)
    
    # Load weights using TensorFlow checkpoint
    print("Loading model weights...")
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint('./')).expect_partial()
    print("Model weights loaded successfully!")
    
    # Generate recommendations
    print(f"\nGenerating top {args.top_n} recommendations for User {args.user_id}...\n")
    recommendations = get_recommendations(
        args.user_id, 
        model, 
        user_encoder, 
        movie_encoder, 
        movies_df, 
        args.top_n,
        args.batch_size
    )
    
    if recommendations is not None:
        print(recommendations.to_string(index=False))
    
if __name__ == "__main__":
    main()
