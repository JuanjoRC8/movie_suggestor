# Arquitectura del Sistema de Recomendación

## Visión General

Este sistema implementa un modelo de **Collaborative Filtering** basado en **Matrix Factorization** usando embeddings de TensorFlow/Keras.

## Componentes Principales

### 1. Data Loader (`data_loader.py`)

**Responsabilidades:**
- Cargar datos desde archivos CSV (ratings.csv, movies.csv)
- Preprocesar datos: codificar IDs de usuarios y películas
- Dividir datos en conjuntos de entrenamiento y validación
- Soportar muestreo para desarrollo rápido
- **Optimizaciones**: Chunked reading, operaciones NumPy vectorizadas

**Funciones clave:**
- `load_data(data_path, sample_frac, use_chunks, chunk_size)`: Carga y preprocesa los datos
- `load_movies(data_path)`: Carga información de películas

**Optimizaciones implementadas:**
- Uso de `np.column_stack` y tipos optimizados (`int32`, `float32`)
- Chunked reading para datasets grandes
- Split manual con NumPy (más rápido que sklearn)

**Salida:**
```python
{
    'X_train': array([[user_id, movie_id], ...]),
    'X_test': array([[user_id, movie_id], ...]),
    'y_train': array([rating, ...]),
    'y_test': array([rating, ...]),
    'n_users': int,
    'n_movies': int,
    'user_encoder': LabelEncoder,
    'movie_encoder': LabelEncoder
}
```

### 2. Modelo (`model.py`)

**Arquitectura: RecommenderNet**

```
Input: [user_id, movie_id]
    ↓
User Embedding (dim=50) ──┐
                          ├─→ Dot Product ──┐
Movie Embedding (dim=50) ─┘                 │
                                            ├─→ Sum ──→ Sigmoid ──→ Output (0-5.5)
User Bias ──────────────────────────────────┤
                                            │
Movie Bias ─────────────────────────────────┘
```

**Capas:**
1. **User Embedding**: Representa cada usuario como un vector de 50 dimensiones
2. **Movie Embedding**: Representa cada película como un vector de 50 dimensiones
3. **User Bias**: Sesgo individual por usuario (escalar)
4. **Movie Bias**: Sesgo individual por película (escalar)
5. **Dot Product**: Captura la interacción usuario-película
6. **Sigmoid**: Normaliza la salida al rango [0, 5.5]

**Regularización:**
- L2 regularization (1e-6) en embeddings para prevenir overfitting

**Inicialización:**
- He Normal initialization para embeddings

### 3. Entrenamiento (`train.py`)

**Proceso:**
1. Cargar datos con `load_data()`
2. Inicializar modelo `RecommenderNet`
3. Compilar con:
   - **Loss**: Mean Squared Error (MSE)
   - **Optimizer**: Adam (lr=0.001)
   - **Metrics**: MAE, RMSE
4. Crear TensorFlow Datasets con prefetching
5. Entrenar con callbacks (EarlyStopping, ReduceLROnPlateau)
6. Guardar:
   - Configuración del modelo (numpy array)
   - Pesos del modelo (TensorFlow checkpoint)
   - Codificadores (pickle)

**Hiperparámetros:**
- `epochs`: Número de épocas (default: 5)
- `batch_size`: Tamaño del batch (default: 1024)
- `embedding_size`: Dimensión de embeddings (default: 50)
- `sample_frac`: Fracción del dataset (default: 0.01)
- `use_mixed_precision`: Mixed precision training (GPU)

**Optimizaciones implementadas:**
- TensorFlow Datasets con `prefetch(AUTOTUNE)`
- Mixed precision training (2-3x más rápido en GPU)
- Early stopping para evitar overfitting
- Learning rate scheduling automático

### 4. Recomendaciones (`recommend.py`)

**Proceso:**
1. Cargar configuración del modelo
2. Cargar codificadores y datos de películas
3. Inicializar modelo con la misma arquitectura
4. Cargar pesos entrenados
5. Para un usuario dado:
   - Generar pares (user, movie) para todas las películas
   - Predecir ratings en batches
   - Usar `argpartition` para top-k (O(n) vs O(n log n))
   - Retornar top-N películas

**Funciones principales:**
```python
get_recommendations(user_id, model, user_encoder, movie_encoder, movies_df, top_n=10, batch_size=10000)
get_batch_recommendations(user_ids, model, ..., n_workers=None)  # Procesamiento paralelo
```

**Optimizaciones implementadas:**
- `np.argpartition` para top-k (4x más rápido)
- Batch predictions para eficiencia de memoria
- Procesamiento paralelo con `ThreadPoolExecutor`
- Operaciones vectorizadas con NumPy

## Flujo de Datos

```
CSV Files (ratings.csv, movies.csv)
    ↓
data_loader.py
    ↓
Preprocessed Data (encoded IDs, train/test split)
    ↓
model.py (RecommenderNet)
    ↓
train.py (training loop)
    ↓
Saved Artifacts (checkpoint, encoders, config)
    ↓
recommend.py (inference)
    ↓
Top-N Recommendations
```

## Métricas de Evaluación

### Mean Squared Error (MSE)
- Mide el error cuadrático promedio entre ratings reales y predichos
- Penaliza errores grandes más que pequeños
- **Objetivo**: Minimizar

### Mean Absolute Error (MAE)
- Mide el error absoluto promedio
- Más interpretable: error promedio en estrellas
- **Objetivo**: Minimizar

### Root Mean Squared Error (RMSE)
- Raíz cuadrada del MSE
- Misma escala que los ratings
- **Objetivo**: Minimizar

## Optimizaciones Implementadas

### Rendimiento:
1. ✅ **TensorFlow Datasets** para pipeline de datos
2. ✅ **Mixed precision training** (GPU)
3. ✅ **Operaciones vectorizadas con NumPy**
4. ✅ **argpartition** para top-k (O(n) vs O(n log n))
5. ✅ **Batch predictions** para eficiencia de memoria
6. ✅ **Procesamiento paralelo** con ThreadPoolExecutor
7. ✅ **Chunked data loading** para datasets grandes
8. ✅ **Early stopping** y learning rate scheduling

Ver `docs/OPTIMIZACIONES.md` para detalles completos.

## Optimizaciones Futuras

### Para Mejorar Precisión:
1. Aumentar `embedding_size` (50 → 100)
2. Entrenar con más datos (`sample_frac` → 1.0)
3. Más épocas de entrenamiento
4. Añadir capas densas adicionales
5. Incorporar features de películas (géneros, año)

### Para Mejorar Velocidad:
1. GPU acceleration para inferencia
2. Caching de embeddings
3. Approximate Nearest Neighbors (FAISS)
4. Model quantization (int8)

### Para Producción:
1. Implementar cache de predicciones
2. Usar TensorFlow Serving
3. Añadir filtrado de películas ya vistas
4. Implementar cold-start para nuevos usuarios
5. A/B testing de diferentes configuraciones

## Referencias

- **Dataset**: MovieLens 32M (https://grouplens.org/datasets/movielens/)
- **Paper**: Harper & Konstan (2015). "The MovieLens Datasets: History and Context"
- **Técnica**: Matrix Factorization via Neural Collaborative Filtering
