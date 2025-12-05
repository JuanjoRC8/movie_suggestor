# Quick Start Guide

## Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar el modelo (1% de datos, ~3 minutos)
python src/train.py --epochs 3 --sample_frac 0.01

# 3. Generar recomendaciones
python src/recommend.py --user_id 1 --top_n 10

# 4. Ver demo completo
python src/demo.py
```

## Comandos Útiles

### Entrenamiento Rápido (para pruebas)
```bash
python src/train.py --epochs 3 --sample_frac 0.01 --batch_size 512
```

### Entrenamiento Completo (dataset completo, ~2-3 horas)
```bash
python src/train.py --epochs 10 --sample_frac 1.0 --batch_size 2048
```

### Recomendaciones para un usuario específico
```bash
python src/recommend.py --user_id 1 --top_n 15
```

### Ver usuarios disponibles
```bash
python -c "import pickle; ue = pickle.load(open('user_encoder.pkl', 'rb')); print('Usuarios disponibles:', ue.classes_[:20])"
```

## Estructura del Proyecto

```
movie_suggestor/
├── ml-32m/                  # Dataset MovieLens 32M
│   ├── ratings.csv          # 32M valoraciones
│   ├── movies.csv           # 87K películas
│   ├── tags.csv             # Tags de usuarios
│   └── links.csv            # Links a IMDb/TMDb
│
├── data_loader.py           # Carga y preprocesamiento
├── model.py                 # Arquitectura del modelo
├── train.py                 # Script de entrenamiento
├── recommend.py             # Generación de recomendaciones
├── demo.py                  # Demo interactivo
│
├── README.md                # Documentación principal
├── ARCHITECTURE.md          # Documentación técnica
├── QUICKSTART.md            # Esta guía
└── requirements.txt         # Dependencias

Archivos generados después del entrenamiento:
├── model_checkpoint-1.*     # Pesos del modelo
├── model_config.npy         # Configuración
├── user_encoder.pkl         # Codificador de usuarios
└── movie_encoder.pkl        # Codificador de películas
```

## Ejemplos de Uso

### Ejemplo 1: Entrenamiento Básico
```bash
# Entrenar con 5% de datos, 5 épocas
python train.py --epochs 5 --sample_frac 0.05

# Salida esperada:
# Loaded 1,600,010 ratings.
# Num users: 112,571, Num movies: 17,748
# Epoch 1/5
# ...
# Final Validation MAE: 0.70
```

### Ejemplo 2: Recomendaciones
```bash
# Obtener 10 recomendaciones para usuario 1
python recommend.py --user_id 1 --top_n 10

# Salida esperada:
# movieId                            title                      genres  predicted_rating
#     110                Braveheart (1995)            Action|Drama|War          3.91
#     593 Silence of the Lambs, The (1991)       Crime|Horror|Thriller          3.88
#     ...
```

### Ejemplo 3: Demo Completo
```bash
python demo.py

# Muestra:
# - Configuración del modelo
# - Recomendaciones para 5 usuarios
# - Estadísticas del sistema
```

## Troubleshooting

### Error: "User ID not found"
**Solución**: El usuario no está en el dataset de entrenamiento. Usa un usuario del conjunto de entrenamiento:
```bash
python -c "import pickle; ue = pickle.load(open('user_encoder.pkl', 'rb')); print(ue.classes_[:10])"
```

### Error: "No module named tensorflow"
**Solución**: Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Error: "model_config.npy not found"
**Solución**: Primero debes entrenar el modelo:
```bash
python train.py --epochs 3 --sample_frac 0.01
```

### Entrenamiento muy lento
**Solución**: 
1. Reduce `sample_frac`: `--sample_frac 0.01`
2. Aumenta `batch_size`: `--batch_size 2048`
3. Reduce épocas: `--epochs 3`

## Próximos Pasos

1. **Experimentar con hiperparámetros**:
   - Prueba diferentes `embedding_size` (50, 100, 150)
   - Ajusta `learning_rate` en `train.py`
   - Aumenta el número de épocas

2. **Mejorar el modelo**:
   - Añade capas densas en `model.py`
   - Incorpora features adicionales (géneros, año)
   - Implementa dropout para regularización

3. **Análisis**:
   - Visualiza embeddings con t-SNE
   - Analiza películas similares
   - Estudia patrones de usuarios

4. **Producción**:
   - Implementa API REST con Flask/FastAPI
   - Añade cache de predicciones
   - Implementa sistema de logging

## Recursos

- **Dataset**: https://grouplens.org/datasets/movielens/
- **Documentación TensorFlow**: https://www.tensorflow.org/
- **Paper original**: Harper & Konstan (2015) - MovieLens Datasets
