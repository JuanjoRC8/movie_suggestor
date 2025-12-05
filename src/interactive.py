"""
Sistema interactivo de recomendaciones basado en preferencias de géneros.
Pregunta al usuario qué géneros le gustan y recomienda películas acordes.
"""

import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict

# Add src to path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import RecommenderNet
from src.data_loader import load_movies
import tensorflow as tf
import pickle

class InteractiveRecommender:
    def __init__(self):
        """Inicializa el sistema de recomendaciones interactivo."""
        self.model = None
        self.user_encoder = None
        self.movie_encoder = None
        self.movies_df = None
        self.all_genres = []
        
    def load_system(self):
        """Carga el modelo y los datos necesarios."""
        print("=" * 80)
        print("🎬 SISTEMA INTERACTIVO DE RECOMENDACIÓN DE PELÍCULAS")
        print("=" * 80)
        print("\n[1/4] Cargando configuración del modelo...")
        
        try:
            config = np.load('model_config.npy')
            num_users = int(config[0])
            num_movies = int(config[1])
            embedding_size = int(config[2])
            print(f"   ✓ Configuración cargada: {num_users:,} usuarios, {num_movies:,} películas")
        except FileNotFoundError:
            print("   ❌ Error: No se encontró el modelo entrenado.")
            print("   Por favor, entrena el modelo primero:")
            print("   python src/train.py --epochs 5 --sample_frac 0.01")
            return False
        
        print("\n[2/4] Cargando codificadores...")
        try:
            with open('user_encoder.pkl', 'rb') as f:
                self.user_encoder = pickle.load(f)
            with open('movie_encoder.pkl', 'rb') as f:
                self.movie_encoder = pickle.load(f)
            print("   ✓ Codificadores cargados")
        except FileNotFoundError:
            print("   ❌ Error: No se encontraron los codificadores.")
            return False
        
        print("\n[3/4] Cargando información de películas...")
        self.movies_df = load_movies("ml-32m")
        print(f"   ✓ {len(self.movies_df):,} películas cargadas")
        
        # Extraer todos los géneros únicos
        genres_set = set()
        for genres_str in self.movies_df['genres'].dropna():
            genres_set.update(genres_str.split('|'))
        self.all_genres = sorted([g for g in genres_set if g != '(no genres listed)'])
        
        print("\n[4/4] Inicializando modelo...")
        self.model = RecommenderNet(num_users, num_movies, embedding_size)
        dummy_input = np.array([[0, 0]], dtype=np.int32)
        _ = self.model(dummy_input)
        
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint('./')).expect_partial()
        print("   ✓ Modelo cargado exitosamente")
        
        return True
    
    def show_genres(self):
        """Muestra los géneros disponibles."""
        print("\n" + "=" * 80)
        print("📋 GÉNEROS DISPONIBLES")
        print("=" * 80)
        
        # Mostrar en columnas
        cols = 3
        for i in range(0, len(self.all_genres), cols):
            row = self.all_genres[i:i+cols]
            print("  " + "".join(f"{idx+i+1:2}. {genre:20}" for idx, genre in enumerate(row)))
        print()
    
    def get_user_preferences(self):
        """Pregunta al usuario por sus preferencias de géneros."""
        self.show_genres()
        
        print("💬 ¿Qué géneros te gustan?")
        print("   Puedes ingresar:")
        print("   - Números separados por comas (ej: 1,5,8)")
        print("   - Nombres de géneros separados por comas (ej: Action,Comedy,Sci-Fi)")
        print("   - 'todos' para ver todas las películas")
        print()
        
        user_input = input("Tu respuesta: ").strip()
        
        if user_input.lower() == 'todos':
            return self.all_genres
        
        selected_genres = []
        
        # Intentar parsear como números
        if any(c.isdigit() for c in user_input):
            try:
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                selected_genres = [self.all_genres[i] for i in indices if 0 <= i < len(self.all_genres)]
            except (ValueError, IndexError):
                print("   ⚠️  Entrada inválida, usando géneros por nombre...")
        
        # Si no funcionó, intentar por nombre
        if not selected_genres:
            input_genres = [g.strip() for g in user_input.split(',')]
            selected_genres = [g for g in input_genres if g in self.all_genres]
        
        if not selected_genres:
            print("   ⚠️  No se reconocieron géneros válidos. Usando Action como default.")
            selected_genres = ['Action']
        
        return selected_genres
    
    def filter_movies_by_genres(self, genres, min_match=1):
        """Filtra películas que contengan al menos min_match de los géneros seleccionados."""
        def matches_genres(movie_genres_str):
            if pd.isna(movie_genres_str):
                return False
            movie_genres = set(movie_genres_str.split('|'))
            matches = len(movie_genres.intersection(genres))
            return matches >= min_match
        
        mask = self.movies_df['genres'].apply(matches_genres)
        filtered_movies = self.movies_df[mask].copy()
        
        # Calcular score de coincidencia
        def genre_score(movie_genres_str):
            movie_genres = set(movie_genres_str.split('|'))
            return len(movie_genres.intersection(genres))
        
        filtered_movies['genre_match_score'] = filtered_movies['genres'].apply(genre_score)
        
        return filtered_movies
    
    def get_recommendations_by_genres(self, genres, top_n=20, batch_size=10000):
        """Genera recomendaciones basadas en géneros preferidos."""
        print(f"\n🔍 Buscando películas de: {', '.join(genres)}")
        
        # Filtrar películas por géneros
        filtered_movies = self.filter_movies_by_genres(genres)
        
        if len(filtered_movies) == 0:
            print("   ❌ No se encontraron películas con esos géneros.")
            return None
        
        print(f"   ✓ Encontradas {len(filtered_movies):,} películas que coinciden")
        
        # Obtener IDs de películas filtradas que están en el modelo
        available_movie_ids = set(self.movie_encoder.classes_)
        filtered_movie_ids = filtered_movies['movieId'].values
        valid_movie_ids = [mid for mid in filtered_movie_ids if mid in available_movie_ids]
        
        if len(valid_movie_ids) == 0:
            print("   ❌ Ninguna de estas películas está en el modelo entrenado.")
            return None
        
        print(f"   ✓ {len(valid_movie_ids):,} películas disponibles en el modelo")
        
        # Crear un "usuario virtual" promedio
        # Usamos el embedding promedio de todos los usuarios
        print("\n🤖 Calculando preferencias basadas en usuarios similares...")
        
        # Predecir ratings para las películas filtradas
        movie_indices = self.movie_encoder.transform(valid_movie_ids)
        
        # Usar el usuario promedio (índice 0 como proxy)
        # En un sistema real, podríamos crear un perfil basado en las preferencias
        user_idx = 0  # Usuario promedio
        
        predictions = []
        for i in range(0, len(movie_indices), batch_size):
            end_idx = min(i + batch_size, len(movie_indices))
            batch_movies = movie_indices[i:end_idx]
            
            user_array = np.repeat(user_idx, len(batch_movies)).astype(np.int32)
            inputs = np.stack([user_array, batch_movies], axis=1)
            
            batch_preds = self.model.predict(inputs, verbose=0).flatten()
            predictions.extend(batch_preds)
        
        predictions = np.array(predictions)
        
        # Combinar predicciones con score de géneros
        # Dar más peso a películas que coinciden con más géneros
        genre_scores = filtered_movies[filtered_movies['movieId'].isin(valid_movie_ids)]['genre_match_score'].values
        
        # Normalizar scores
        normalized_genre_scores = genre_scores / max(genre_scores)
        
        # Score final: 70% predicción del modelo + 30% coincidencia de géneros
        final_scores = 0.7 * predictions + 0.3 * normalized_genre_scores * 5.0
        
        # Obtener top N
        if top_n < len(final_scores):
            top_indices = np.argpartition(final_scores, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(final_scores)[::-1]
        
        # Crear DataFrame de resultados
        recommended_movie_ids = [valid_movie_ids[i] for i in top_indices]
        recommended_scores = final_scores[top_indices]
        
        results = filtered_movies[filtered_movies['movieId'].isin(recommended_movie_ids)].copy()
        score_map = dict(zip(recommended_movie_ids, recommended_scores))
        results['predicted_rating'] = results['movieId'].map(score_map)
        results = results.sort_values('predicted_rating', ascending=False)
        
        return results[['movieId', 'title', 'genres', 'predicted_rating', 'genre_match_score']]
    
    def run(self):
        """Ejecuta el sistema interactivo."""
        if not self.load_system():
            return
        
        print("\n" + "=" * 80)
        print("✨ ¡Sistema listo!")
        print("=" * 80)
        
        while True:
            # Obtener preferencias del usuario
            selected_genres = self.get_user_preferences()
            
            print(f"\n✅ Géneros seleccionados: {', '.join(selected_genres)}")
            
            # Preguntar cuántas recomendaciones quiere
            try:
                num_recs = input("\n¿Cuántas recomendaciones quieres? (default: 10): ").strip()
                num_recs = int(num_recs) if num_recs else 10
            except ValueError:
                num_recs = 10
            
            # Generar recomendaciones
            print("\n" + "=" * 80)
            print(f"🎬 GENERANDO {num_recs} RECOMENDACIONES")
            print("=" * 80)
            
            recommendations = self.get_recommendations_by_genres(selected_genres, top_n=num_recs)
            
            if recommendations is not None and len(recommendations) > 0:
                print(f"\n🌟 Top {len(recommendations)} películas recomendadas:\n")
                
                # Mostrar resultados formateados
                for idx, row in enumerate(recommendations.itertuples(), 1):
                    rating_stars = "⭐" * int(round(row.predicted_rating))
                    genre_match = "🎯" * row.genre_match_score
                    
                    print(f"{idx:2}. {row.title}")
                    print(f"    Géneros: {row.genres}")
                    print(f"    Rating predicho: {row.predicted_rating:.2f} {rating_stars}")
                    print(f"    Coincidencia de géneros: {genre_match} ({row.genre_match_score})")
                    print()
            
            # Preguntar si quiere más recomendaciones
            print("=" * 80)
            continue_choice = input("\n¿Quieres más recomendaciones? (s/n): ").strip().lower()
            
            if continue_choice not in ['s', 'si', 'sí', 'y', 'yes']:
                print("\n👋 ¡Gracias por usar el sistema de recomendaciones!")
                print("   ¡Disfruta de tus películas! 🍿")
                break
            
            print("\n" + "=" * 80)

def main():
    recommender = InteractiveRecommender()
    try:
        recommender.run()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
