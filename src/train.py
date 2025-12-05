import tensorflow as tf
import numpy as np
import sys
import os

# Add src to path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data
from src.model import RecommenderNet
import argparse
import matplotlib.pyplot as plt

def train(epochs=5, batch_size=1024, embedding_size=50, sample_frac=0.1):
    # Ensure we are in the right directory or point to the right place
    data_path = "ml-32m"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = load_data(data_path, sample_frac=sample_frac)
    
    num_users = data['n_users']
    num_movies = data['n_movies']
    
    print(f"Initializing model with {num_users} users and {num_movies} movies...")
    model = RecommenderNet(num_users, num_movies, embedding_size)
    
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    
    print("Starting training...")
    history = model.fit(
        x=data['X_train'],
        y=data['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(data['X_test'], data['y_test']),
        verbose=1
    )
    
    
    # Save model configuration using numpy
    import pickle
    
    config = np.array([num_users, num_movies, embedding_size])
    np.save('model_config.npy', config)
    print("Model configuration saved to model_config.npy")
    
    # Save model weights using TensorFlow checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = checkpoint.save('model_checkpoint')
    print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Save encoders for inference
    with open('user_encoder.pkl', 'wb') as f:
        pickle.dump(data['user_encoder'], f)
    with open('movie_encoder.pkl', 'wb') as f:
        pickle.dump(data['movie_encoder'], f)
    print("Encoders saved.")
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Training MAE: {history.history['mean_absolute_error'][-1]:.4f}")
    print(f"Final Training RMSE: {history.history['root_mean_squared_error'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Validation MAE: {history.history['val_mean_absolute_error'][-1]:.4f}")
    print(f"Final Validation RMSE: {history.history['val_root_mean_squared_error'][-1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Movie Recommender')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=50, help='Embedding size')
    parser.add_argument('--sample_frac', type=float, default=0.01, help='Fraction of data to use (0.0-1.0)')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_size=args.embedding_size,
        sample_frac=args.sample_frac
    )
