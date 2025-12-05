"""
Script para analizar y visualizar estadísticas del sistema de recomendaciones.
"""

import numpy as np
import pandas as pd
import pickle
from collections import Counter

def analyze_dataset():
    """Analiza el dataset de entrenamiento."""
    print("=" * 80)
    print("ANÁLISIS DEL DATASET")
    print("=" * 80)
    
    # Load encoders
    with open('user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open('movie_encoder.pkl', 'rb') as f:
        movie_encoder = pickle.load(f)
    
    # Load movies
    movies_df = pd.read_csv('ml-32m/movies.csv')
    
    print(f"\n📊 Estadísticas Generales:")
    print(f"   • Usuarios en el modelo: {len(user_encoder.classes_):,}")
    print(f"   • Películas en el modelo: {len(movie_encoder.classes_):,}")
    print(f"   • Total de películas en dataset: {len(movies_df):,}")
    
    # Analyze genres
    print(f"\n🎬 Análisis de Géneros:")
    all_genres = []
    for genres_str in movies_df['genres']:
        if genres_str != '(no genres listed)':
            genres = genres_str.split('|')
            all_genres.extend(genres)
    
    genre_counts = Counter(all_genres)
    print(f"   • Total de géneros únicos: {len(genre_counts)}")
    print(f"\n   Top 10 géneros más comunes:")
    for genre, count in genre_counts.most_common(10):
        print(f"      {genre:20s}: {count:5,} películas")
    
    # Analyze movies in model
    movies_in_model = movies_df[movies_df['movieId'].isin(movie_encoder.classes_)]
    print(f"\n🎯 Películas en el Modelo:")
    print(f"   • Cobertura: {len(movies_in_model)/len(movies_df)*100:.1f}%")
    
    # Sample movies
    print(f"\n🎥 Muestra de Películas en el Modelo:")
    sample_movies = movies_in_model.sample(min(10, len(movies_in_model)))
    for _, row in sample_movies.iterrows():
        print(f"   • {row['title']:50s} [{row['genres']}]")
    
    # Model configuration
    config = np.load('model_config.npy')
    num_users = int(config[0])
    num_movies = int(config[1])
    embedding_size = int(config[2])
    
    print(f"\n⚙️  Configuración del Modelo:")
    print(f"   • Dimensión de embeddings: {embedding_size}")
    print(f"   • Parámetros totales: ~{(num_users + num_movies) * (embedding_size + 1) * 2:,}")
    print(f"   • User embeddings: {num_users:,} × {embedding_size} = {num_users * embedding_size:,}")
    print(f"   • Movie embeddings: {num_movies:,} × {embedding_size} = {num_movies * embedding_size:,}")
    print(f"   • Biases: {num_users + num_movies:,}")
    
    # Recommendations statistics
    print(f"\n📈 Capacidad del Sistema:")
    total_predictions = num_users * num_movies
    print(f"   • Predicciones posibles: {total_predictions:,}")
    print(f"   • Espacio de búsqueda: {total_predictions / 1e9:.2f} mil millones")
    
    print("\n" + "=" * 80)

def show_popular_movies():
    """Muestra las películas más populares del dataset."""
    print("\n" + "=" * 80)
    print("PELÍCULAS MÁS POPULARES (por número de valoraciones)")
    print("=" * 80)
    
    # Load ratings
    print("\nCargando ratings... (esto puede tomar un momento)")
    ratings_df = pd.read_csv('ml-32m/ratings.csv')
    movies_df = pd.read_csv('ml-32m/movies.csv')
    
    # Count ratings per movie
    movie_counts = ratings_df['movieId'].value_counts().head(20)
    
    print(f"\nTop 20 películas más valoradas:\n")
    for i, (movie_id, count) in enumerate(movie_counts.items(), 1):
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            genres = movie_info.iloc[0]['genres']
            avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
            print(f"{i:2d}. {title:50s}")
            print(f"    Valoraciones: {count:7,} | Rating promedio: {avg_rating:.2f} | Géneros: {genres}")
            print()
    
    print("=" * 80)

def main():
    import sys
    
    print("\n🎬 Sistema de Análisis de Recomendaciones de Películas\n")
    
    try:
        analyze_dataset()
        
        # Ask if user wants to see popular movies
        if len(sys.argv) > 1 and sys.argv[1] == '--popular':
            show_popular_movies()
        else:
            print("\n💡 Tip: Ejecuta 'python analyze.py --popular' para ver las películas más populares")
            print("   (Advertencia: esto cargará el dataset completo de ratings)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Archivo no encontrado - {e}")
        print("   Asegúrate de haber entrenado el modelo primero:")
        print("   python train.py --epochs 3 --sample_frac 0.01")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
