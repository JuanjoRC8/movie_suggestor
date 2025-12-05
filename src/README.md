# Source Code

Este directorio contiene todo el código fuente del sistema de recomendaciones.

## Archivos

- **`data_loader.py`** - Carga y preprocesamiento de datos
- **`model.py`** - Arquitectura del modelo (RecommenderNet)
- **`train.py`** - Script de entrenamiento
- **`recommend.py`** - Generación de recomendaciones
- **`demo.py`** - Demo interactivo del sistema
- **`analyze.py`** - Análisis de estadísticas del dataset

## Uso

Todos los scripts deben ejecutarse desde el directorio raíz del proyecto:

```bash
# Entrenar el modelo
python src/train.py --epochs 3 --sample_frac 0.01

# Generar recomendaciones
python src/recommend.py --user_id 1 --top_n 10

# Ver demo
python src/demo.py

# Analizar dataset
python src/analyze.py
```
