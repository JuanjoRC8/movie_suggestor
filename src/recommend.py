import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os

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

def get_recommendations(user_id, model, user_encoder, movie_encoder, movies_df, top_n=10):
    """
    Generate top N movie recommendations for a given user.
    
    Args:
        user_id (int): Original user ID from the dataset.
        model: Trained RecommenderNet model.
        user_encoder: LabelEncoder for users.
        movie_encoder: LabelEncoder for movies.
        movies_df: DataFrame containing movie information.
        top_n (int): Number of recommendations to return.
        
    Returns:
        DataFrame with recommended movies.
    """
    # Encode the user ID
    try:
        user_encoded = user_encoder.transform([user_id])[0]
    except ValueError:
        print(f"User ID {user_id} not found in training data.")
        return None
    
    # Get all movie IDs
    all_movie_ids = movie_encoder.classes_
    all_movie_encoded = np.arange(len(all_movie_ids))
    
    # Create input pairs (user, movie) for all movies
    user_array = np.full(len(all_movie_encoded), user_encoded)
    movie_array = all_movie_encoded
    
    # Stack into input format
    inputs = np.column_stack([user_array, movie_array])
    
    # Predict ratings
    predictions = model.predict(inputs, verbose=0).flatten()
    
    # Get top N movie indices
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    # Get the original movie IDs
    recommended_movie_ids = all_movie_ids[top_indices]
    predicted_ratings = predictions[top_indices]
    
    # Get movie details
    recommendations = movies_df[movies_df['movieId'].isin(recommended_movie_ids)].copy()
    recommendations['predicted_rating'] = predicted_ratings
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    return recommendations[['movieId', 'title', 'genres', 'predicted_rating']]

def main():
    parser = argparse.ArgumentParser(description='Generate Movie Recommendations')
    parser.add_argument('--user_id', type=int, required=True, help='User ID to generate recommendations for')
    parser.add_argument('--top_n', type=int, default=10, help='Number of recommendations')
    
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
    
    print(f"Initializing model ({num_users} users, {num_movies} movies, embedding_size={embedding_size})...")
    model = RecommenderNet(num_users, num_movies, embedding_size)
    
    # Build the model by calling it once
    dummy_input = np.array([[0, 0]])
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
        args.top_n
    )
    
    if recommendations is not None:
        print(recommendations.to_string(index=False))
    
if __name__ == "__main__":
    main()
