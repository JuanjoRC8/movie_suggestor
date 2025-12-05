# Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de recomendación de películas basado en **Collaborative Filtering** usando TensorFlow y el dataset MovieLens 32M.

## 🚀 Quick Start

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/movie_suggestor.git
cd movie_suggestor

# 2. Descargar el dataset (ver docs/DATASET.md)
# El dataset NO está incluido en el repo debido a su tamaño

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar el modelo
python src/train.py --epochs 3 --sample_frac 0.01

# 5. Generar recomendaciones
python src/recommend.py --user_id 1 --top_n 10
```

## Descripción

Este sistema utiliza una arquitectura de **Matrix Factorization** implementada con embeddings de TensorFlow para predecir las valoraciones que un usuario daría a películas que no ha visto, y recomendar las películas con mayor valoración predicha.

### Arquitectura del Modelo

- **User Embedding**: Representación latente de cada usuario
- **Movie Embedding**: Representación latente de cada película
- **Dot Product**: Captura la interacción usuario-película
- **Biases**: Sesgo individual para usuarios y películas
- **Sigmoid Activation**: Normaliza la salida al rango de valoraciones (0.5-5.0)

### Métricas de Evaluación

- **Loss**: Mean Squared Error (MSE)
- **Métricas**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

## Dataset

El dataset MovieLens 32M contiene:
- **32,000,204** valoraciones
- **87,585** películas
- **200,948** usuarios
- Valoraciones en escala de 0.5 a 5.0 estrellas

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### 1. Entrenar el Modelo

```bash
# Entrenar con 1% de los datos (para pruebas rápidas)
python src/train.py --epochs 5 --sample_frac 0.01

# Entrenar con el dataset completo
python src/train.py --epochs 10 --sample_frac 1.0 --batch_size 2048

# Opciones disponibles:
# --epochs: Número de épocas (default: 5)
# --batch_size: Tamaño del batch (default: 1024)
# --embedding_size: Dimensión de los embeddings (default: 50)
# --sample_frac: Fracción del dataset a usar (default: 0.01)
```

El entrenamiento guardará:
- `model_checkpoint-*`: Pesos del modelo entrenado (TensorFlow checkpoint)
- `model_config.npy`: Configuración del modelo (numpy array)
- `user_encoder.pkl`: Codificador de IDs de usuario
- `movie_encoder.pkl`: Codificador de IDs de película

### 2. Generar Recomendaciones

```bash
# Obtener 10 recomendaciones para el usuario 1
python src/recommend.py --user_id 1 --top_n 10

# Opciones disponibles:
# --user_id: ID del usuario (requerido)
# --top_n: Número de recomendaciones (default: 10)
```

## Estructura del Proyecto

```
movie_suggestor/
├── src/                     # Código fuente
│   ├── data_loader.py       # Carga y preprocesamiento de datos
│   ├── model.py             # Arquitectura del modelo (RecommenderNet)
│   ├── train.py             # Script de entrenamiento
│   ├── recommend.py         # Generación de recomendaciones
│   ├── demo.py              # Demo interactivo
│   └── analyze.py           # Análisis de estadísticas
│
├── docs/                    # Documentación
│   ├── ARCHITECTURE.md      # Documentación técnica
│   ├── QUICKSTART.md        # Guía de inicio rápido
│   └── DATASET.md           # Instrucciones de descarga del dataset
│
├── ml-32m/                  # Dataset MovieLens 32M
│   ├── README.txt           # Documentación del dataset
│   ├── checksums.txt        # Checksums MD5
│   ├── movies.csv           # Información de películas (incluido)
│   ├── ratings.csv          # Valoraciones (DESCARGAR)
│   ├── tags.csv             # Tags (DESCARGAR)
│   └── links.csv            # Enlaces IMDb/TMDb (DESCARGAR)
│
├── README.md                # Este archivo
├── LICENSE                  # Licencia MIT
├── requirements.txt         # Dependencias Python
└── .gitignore               # Archivos ignorados por Git
```

## Ejemplo de Salida

```
Generating top 10 recommendations for User 1...

movieId                            title                      genres  predicted_rating
    110                Braveheart (1995)            Action|Drama|War          3.91
    593 Silence of the Lambs, The (1991)       Crime|Horror|Thriller          3.88
    608                     Fargo (1996) Comedy|Crime|Drama|Thriller          3.84
   1203              12 Angry Men (1957)                       Drama          3.83
   2028       Saving Private Ryan (1998)            Action|Drama|War          3.72
   2858           American Beauty (1999)               Drama|Romance          3.71
   2959                Fight Club (1999) Action|Crime|Drama|Thriller          3.71
   4226                   Memento (2000)            Mystery|Thriller          3.70
  48516             Departed, The (2006)        Crime|Drama|Thriller          3.67
  58559          Dark Knight, The (2008)     Action|Crime|Drama|IMAX          3.67
```

## Métricas de Entrenamiento

Con el 1% del dataset (320,000 valoraciones aprox.):
- **Training Loss (MSE)**: ~0.75
- **Training MAE**: ~0.68
- **Training RMSE**: ~0.87
- **Validation Loss (MSE)**: ~0.78
- **Validation MAE**: ~0.70
- **Validation RMSE**: ~0.88

Estos resultados indican que el modelo predice valoraciones con un error promedio de ~0.7 estrellas.

## Archivos Generados

Después del entrenamiento, se generan los siguientes archivos:
- `model_checkpoint-1.data-*`: Pesos del modelo (TensorFlow checkpoint)
- `model_checkpoint-1.index`: Índice del checkpoint
- `model_config.npy`: Configuración del modelo (numpy array)
- `user_encoder.pkl`: Codificador de IDs de usuario
- `movie_encoder.pkl`: Codificador de IDs de película

## Notas

- Para entrenamientos con el dataset completo, se recomienda usar GPU
- El modelo usa regularización L2 para prevenir overfitting
- Los embeddings se inicializan con He normal initialization
