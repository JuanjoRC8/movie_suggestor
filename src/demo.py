"""
Script de ejemplo para demostrar el sistema de recomendaciones con optimizaciones.
Muestra recomendaciones para varios usuarios y analiza los resultados.
"""

import pickle
import numpy as np
import pandas as pd
import sys
import os
import time

# Add src to path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.recommend import get_recommendations, load_encoders, get_batch_recommendations
from src.data_loader import load_movies
from src.model import RecommenderNet
import tensorflow as tf

def main():
    print("=" * 80)
    print("DEMO: Sistema de Recomendación de Películas (Optimizado)")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/4] Cargando configuración del modelo...")
    config = np.load('model_config.npy')
    num_users = int(config[0])
    num_movies = int(config[1])
    embedding_size = int(config[2])
    print(f"   ✓ Usuarios: {num_users:,}, Películas: {num_movies:,}, Embedding: {embedding_size}")
    
    # Load encoders
    print("\n[2/4] Cargando codificadores...")
    user_encoder, movie_encoder = load_encoders()
    print(f"   ✓ Codificadores cargados")
    
    # Load movies
    print("\n[3/4] Cargando información de películas...")
    movies_df = load_movies("ml-32m")
    print(f"   ✓ {len(movies_df):,} películas cargadas")
    
    # Initialize and load model
    print("\n[4/4] Inicializando modelo...")
    model = RecommenderNet(num_users, num_movies, embedding_size)
    dummy_input = np.array([[0, 0]], dtype=np.int32)
    _ = model(dummy_input)
    
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint('./')).expect_partial()
    print("   ✓ Modelo cargado exitosamente")
    
    # Get sample users
    sample_users = user_encoder.classes_[:5]
    
    print("\n" + "=" * 80)
    print("RECOMENDACIONES INDIVIDUALES (Modo Secuencial)")
    print("=" * 80)
    
    total_time = 0
    for user_id in sample_users:
        print(f"\n{'─' * 80}")
        print(f"Usuario ID: {user_id}")
        print(f"{'─' * 80}")
        
        start_time = time.time()
        recommendations = get_recommendations(
            user_id, 
            model, 
            user_encoder, 
            movie_encoder, 
            movies_df, 
            top_n=5,
            batch_size=10000
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if recommendations is not None:
            print(recommendations.to_string(index=False))
            print(f"\n⏱️  Tiempo: {elapsed:.3f}s")
        
        print()
    
    print(f"Tiempo total (secuencial): {total_time:.3f}s")
    print(f"Tiempo promedio por usuario: {total_time/len(sample_users):.3f}s")
    
    # Demonstrate parallel processing
    print("\n" + "=" * 80)
    print("RECOMENDACIONES EN PARALELO (Modo Optimizado)")
    print("=" * 80)
    
    print(f"\nGenerando recomendaciones para {len(sample_users)} usuarios en paralelo...")
    start_time = time.time()
    batch_results = get_batch_recommendations(
        sample_users,
        model,
        user_encoder,
        movie_encoder,
        movies_df,
        top_n=5,
        n_workers=min(4, len(sample_users))
    )
    parallel_time = time.time() - start_time
    
    print(f"\n⏱️  Tiempo total (paralelo): {parallel_time:.3f}s")
    print(f"⚡ Speedup: {total_time/parallel_time:.2f}x más rápido")
    
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DEL SISTEMA")
    print("=" * 80)
    print(f"Total de usuarios en el modelo:     {num_users:,}")
    print(f"Total de películas en el modelo:    {num_movies:,}")
    print(f"Dimensión de embeddings:            {embedding_size}")
    print(f"Parámetros del modelo:              ~{(num_users + num_movies) * (embedding_size + 1) * 2:,}")
    print(f"\nOptimizaciones aplicadas:")
    print(f"  ✓ Operaciones vectorizadas con NumPy")
    print(f"  ✓ Procesamiento en paralelo con ThreadPoolExecutor")
    print(f"  ✓ Batch predictions para eficiencia de memoria")
    print(f"  ✓ argpartition para top-k (O(n) vs O(n log n))")
    print("=" * 80)

if __name__ == "__main__":
    main()
