import tensorflow as tf
import numpy as np
import sys
import os

# Add src to path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data
from src.model import RecommenderNet
import argparse

def train(epochs=5, batch_size=1024, embedding_size=50, sample_frac=0.1, use_mixed_precision=False):
    """
    Train the recommendation model with optimized settings.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        embedding_size (int): Dimension of embeddings.
        sample_frac (float): Fraction of dataset to use.
        use_mixed_precision (bool): Use mixed precision training for faster computation.
    """
    # Enable mixed precision if requested (faster on modern GPUs)
    if use_mixed_precision:
        print("Enabling mixed precision training...")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Ensure we are in the right directory or point to the right place
    data_path = "ml-32m"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load data with optimized numpy operations
    use_chunks = sample_frac >= 0.5  # Use chunked reading for large datasets
    data = load_data(data_path, sample_frac=sample_frac, use_chunks=use_chunks)
    
    num_users = data['n_users']
    num_movies = data['n_movies']
    
    print(f"\nInitializing model with {num_users:,} users and {num_movies:,} movies...")
    model = RecommenderNet(num_users, num_movies, embedding_size)
    
    # Compile with optimized settings
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    
    # Create TensorFlow datasets for better performance
    print("\nCreating TensorFlow datasets for optimized training...")
    
    # Convert numpy arrays to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((data['X_train'], data['y_train']))
    train_dataset = train_dataset.shuffle(buffer_size=min(100000, len(data['X_train'])))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for better performance
    
    test_dataset = tf.data.Dataset.from_tensor_slices((data['X_test'], data['y_test']))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"\nStarting training with batch_size={batch_size}, epochs={epochs}...")
    print(f"Training samples: {len(data['X_train']):,}, Validation samples: {len(data['X_test']):,}")
    
    # Train with callbacks for better monitoring
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model configuration using numpy
    import pickle
    
    config = np.array([num_users, num_movies, embedding_size], dtype=np.int32)
    np.save('model_config.npy', config)
    print("\nModel configuration saved to model_config.npy")
    
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
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Final Training Loss (MSE):  {history.history['loss'][-1]:.4f}")
    print(f"Final Training MAE:         {history.history['mean_absolute_error'][-1]:.4f}")
    print(f"Final Training RMSE:        {history.history['root_mean_squared_error'][-1]:.4f}")
    print(f"Final Validation Loss (MSE):{history.history['val_loss'][-1]:.4f}")
    print(f"Final Validation MAE:       {history.history['val_mean_absolute_error'][-1]:.4f}")
    print(f"Final Validation RMSE:      {history.history['val_root_mean_squared_error'][-1]:.4f}")
    print("="*80)
    
    # Print performance tips
    print("\n💡 Performance Tips:")
    print(f"   • Current batch size: {batch_size}")
    print(f"   • Try larger batch sizes (2048, 4096) for faster training")
    print(f"   • Use --use_mixed_precision for GPU acceleration")
    print(f"   • Increase --sample_frac to use more data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Movie Recommender')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=50, help='Embedding size')
    parser.add_argument('--sample_frac', type=float, default=0.01, help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_size=args.embedding_size,
        sample_frac=args.sample_frac,
        use_mixed_precision=args.use_mixed_precision
    )
