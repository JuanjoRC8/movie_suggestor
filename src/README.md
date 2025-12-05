# Source Code

Este directorio contiene todo el código fuente del sistema de recomendaciones con optimizaciones de rendimiento.

## Archivos

- **`data_loader.py`** - Carga y preprocesamiento de datos (optimizado con NumPy)
- **`model.py`** - Arquitectura del modelo (RecommenderNet)
- **`train.py`** - Script de entrenamiento (con tf.data y mixed precision)
- **`recommend.py`** - Generación de recomendaciones (vectorizado y paralelo)
- **`demo.py`** - Demo interactivo con benchmarking
- **`analyze.py`** - Análisis de estadísticas del dataset
- **`interactive.py`** - Sistema interactivo de recomendaciones por géneros

## Optimizaciones Implementadas

- ✅ **Operaciones vectorizadas con NumPy** (2-4x más rápido)
- ✅ **Procesamiento paralelo** con ThreadPoolExecutor
- ✅ **TensorFlow Datasets** con prefetching
- ✅ **Mixed precision training** (GPU)
- ✅ **argpartition** para top-k (O(n) vs O(n log n))
- ✅ **Batch predictions** para eficiencia de memoria

## Uso

Todos los scripts deben ejecutarse desde el directorio raíz del proyecto:

```bash
# Entrenar el modelo (básico)
python src/train.py --epochs 3 --sample_frac 0.01

# Entrenar con mixed precision (GPU)
python src/train.py --epochs 10 --sample_frac 1.0 --batch_size 4096 --use_mixed_precision

# Generar recomendaciones
python src/recommend.py --user_id 1 --top_n 10

# Recomendaciones optimizadas
python src/recommend.py --user_id 1 --top_n 20 --batch_size 20000

# Ver demo con procesamiento paralelo
python src/demo.py

# Sistema interactivo por géneros
python src/interactive.py

# Analizar dataset
python src/analyze.py
```

Ver `docs/OPTIMIZACIONES.md` para detalles completos de las optimizaciones.
