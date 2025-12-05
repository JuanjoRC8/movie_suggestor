"""
Script de ejemplo para demostrar el sistema de recomendaciones.
Muestra recomendaciones para varios usuarios y analiza los resultados.
"""

import pickle
import numpy as np
import pandas as pd
import sys
import os

# Add src to path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.recommend import get_recommendations, load_encoders
from src.data_loader import load_movies
from src.model import RecommenderNet
import tensorflow as tf

def main():
    print("=" * 80)
    print("DEMO: Sistema de Recomendación de Películas")
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
    dummy_input = np.array([[0, 0]])
    _ = model(dummy_input)
    
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint('./')).expect_partial()
    print("   ✓ Modelo cargado exitosamente")
    
    # Get sample users
    sample_users = user_encoder.classes_[:5]
    
    print("\n" + "=" * 80)
    print("RECOMENDACIONES PARA USUARIOS DE EJEMPLO")
    print("=" * 80)
    
    for user_id in sample_users:
        print(f"\n{'─' * 80}")
        print(f"Usuario ID: {user_id}")
        print(f"{'─' * 80}")
        
        recommendations = get_recommendations(
            user_id, 
            model, 
            user_encoder, 
            movie_encoder, 
            movies_df, 
            top_n=5
        )
        
        if recommendations is not None:
            print(recommendations.to_string(index=False))
        
        print()
    
    print("=" * 80)
    print("ESTADÍSTICAS DEL SISTEMA")
    print("=" * 80)
    print(f"Total de usuarios en el modelo: {num_users:,}")
    print(f"Total de películas en el modelo: {num_movies:,}")
    print(f"Dimensión de embeddings: {embedding_size}")
    print(f"Parámetros del modelo: ~{(num_users + num_movies) * embedding_size * 2:,}")
    print("=" * 80)

if __name__ == "__main__":
    main()
