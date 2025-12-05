# Guía de Uso: Sistema Interactivo de Recomendaciones

## 🎬 ¿Qué es el Sistema Interactivo?

El sistema interactivo (`interactive.py`) es una herramienta que te permite obtener recomendaciones de películas basadas en tus géneros favoritos, sin necesidad de conocer IDs de usuario.

## 🚀 Inicio Rápido

```bash
python src/interactive.py
```

## 📋 Cómo Funciona

### 1. **Selección de Géneros**

El sistema te mostrará 19 géneros disponibles:

```
📋 GÉNEROS DISPONIBLES
================================================================================
  1. Action               2. Adventure            3. Animation
  4. Children             5. Comedy               6. Crime
  7. Documentary          8. Drama                9. Fantasy
 10. Film-Noir           11. Horror              12. IMAX
 13. Musical             14. Mystery             15. Romance
 16. Sci-Fi              17. Thriller            18. War
 19. Western
```

### 2. **Formas de Seleccionar**

Puedes elegir géneros de tres maneras:

#### **Opción A: Por Números**
```
Tu respuesta: 1,5,8
```
Selecciona: Action, Comedy, Drama

#### **Opción B: Por Nombres**
```
Tu respuesta: Action,Comedy,Drama
```

#### **Opción C: Todos los Géneros**
```
Tu respuesta: todos
```

### 3. **Número de Recomendaciones**

```
¿Cuántas recomendaciones quieres? (default: 10): 15
```

Puedes pedir entre 1 y cualquier número de recomendaciones.

### 4. **Resultados**

El sistema te mostrará:

```
🌟 Top 15 películas recomendadas:

 1. Children of Men (2006)
    Géneros: Action|Adventure|Drama|Sci-Fi|Thriller
    Rating predicho: 4.10 ⭐⭐⭐⭐
    Coincidencia de géneros: 🎯🎯🎯 (3)

 2. Matrix, The (1999)
    Géneros: Action|Sci-Fi|Thriller
    Rating predicho: 3.96 ⭐⭐⭐⭐
    Coincidencia de géneros: 🎯🎯🎯 (3)
```

**Explicación de los indicadores:**
- ⭐ **Estrellas**: Rating predicho por el modelo
- 🎯 **Dardos**: Cuántos de tus géneros coinciden
- **Número entre paréntesis**: Score de coincidencia

## 🎯 Ejemplos de Uso

### Ejemplo 1: Fan de Acción y Ciencia Ficción

```bash
$ python src/interactive.py

Tu respuesta: 1,16,17  # Action, Sci-Fi, Thriller
¿Cuántas recomendaciones quieres?: 10

# Resultados:
# - Matrix, The (1999)
# - Inception (2010)
# - V for Vendetta (2006)
# - Equilibrium (2002)
# ...
```

### Ejemplo 2: Amante de Comedias Románticas

```bash
Tu respuesta: Comedy,Romance
¿Cuántas recomendaciones quieres?: 15

# Resultados:
# - When Harry Met Sally... (1989)
# - Sleepless in Seattle (1993)
# - Notting Hill (1999)
# ...
```

### Ejemplo 3: Explorador de Géneros

```bash
Tu respuesta: 7,10,14  # Documentary, Film-Noir, Mystery
¿Cuántas recomendaciones quieres?: 20

# Descubre películas únicas en géneros menos comunes
```

## 🔧 Cómo Funciona Internamente

### Algoritmo de Recomendación

El sistema combina dos enfoques:

1. **Predicción del Modelo (70%)**
   - Usa el modelo entrenado de Collaborative Filtering
   - Predice qué rating darías a cada película

2. **Coincidencia de Géneros (30%)**
   - Calcula cuántos de tus géneros tiene cada película
   - Da más peso a películas con más coincidencias

**Fórmula:**
```
Score Final = 0.7 × Predicción del Modelo + 0.3 × Score de Géneros
```

### Filtrado de Películas

1. **Filtro inicial**: Solo películas que contengan al menos 1 de tus géneros
2. **Disponibilidad**: Solo películas en el modelo entrenado
3. **Ordenamiento**: Por score final (predicción + géneros)
4. **Top-N**: Selecciona las mejores N películas

## 💡 Tips y Trucos

### Para Mejores Resultados

1. **Sé específico**: Selecciona 2-4 géneros relacionados
   ```
   Bueno: Action,Sci-Fi,Thriller
   Menos específico: Action,Comedy,Horror,Romance
   ```

2. **Experimenta**: Prueba combinaciones inusuales
   ```
   Animation,Sci-Fi
   Musical,Crime
   Documentary,Thriller
   ```

3. **Ajusta la cantidad**: 
   - 5-10 recomendaciones: Lo mejor de lo mejor
   - 15-20 recomendaciones: Más opciones para explorar
   - 30+: Descubrimiento profundo

### Casos de Uso

#### **Noche de Películas**
```
Tu respuesta: Action,Adventure
¿Cuántas?: 5
```
Obtén las 5 mejores para elegir rápido.

#### **Maratón de Fin de Semana**
```
Tu respuesta: Sci-Fi,Thriller
¿Cuántas?: 20
```
Planifica un maratón completo.

#### **Descubrimiento**
```
Tu respuesta: Film-Noir,Mystery
¿Cuántas?: 30
```
Explora géneros menos conocidos.

## 🔄 Sesión Continua

Después de cada recomendación:

```
¿Quieres más recomendaciones? (s/n): s
```

- **s/sí/si/y/yes**: Nueva ronda de recomendaciones
- **n/no**: Salir del sistema

Esto te permite:
- Probar diferentes combinaciones de géneros
- Ajustar el número de recomendaciones
- Explorar sin reiniciar el programa

## 🎨 Personalización Futura

El sistema está diseñado para ser extensible. Futuras mejoras podrían incluir:

- ✨ Filtrado por año de lanzamiento
- ✨ Exclusión de películas ya vistas
- ✨ Guardar favoritos
- ✨ Exportar lista a archivo
- ✨ Integración con servicios de streaming

## ⚠️ Notas Importantes

### Limitaciones

1. **Modelo Entrenado Requerido**
   ```bash
   # Si no has entrenado el modelo:
   python src/train.py --epochs 5 --sample_frac 0.01
   ```

2. **Disponibilidad de Películas**
   - Solo películas en el dataset MovieLens 32M
   - Solo películas en el conjunto de entrenamiento

3. **Géneros del Dataset**
   - Los géneros son los definidos por MovieLens
   - Algunas películas pueden tener múltiples géneros

### Solución de Problemas

#### Error: "No se encontró el modelo entrenado"
```bash
# Solución: Entrena el modelo primero
python src/train.py --epochs 5 --sample_frac 0.01
```

#### Error: "No se encontraron películas con esos géneros"
```bash
# Solución: Prueba con géneros más comunes
# Géneros populares: Action, Drama, Comedy, Thriller
```

#### Las recomendaciones no son buenas
```bash
# Solución: Entrena con más datos
python src/train.py --epochs 10 --sample_frac 0.1
```

## 📊 Comparación con Otros Métodos

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Interactive** | ✅ No necesitas user ID<br>✅ Basado en tus gustos<br>✅ Fácil de usar | ⚠️ No personalizado a tu historial |
| **recommend.py** | ✅ Personalizado a usuario<br>✅ Usa todo el historial | ⚠️ Necesitas user ID<br>⚠️ Solo usuarios en dataset |
| **demo.py** | ✅ Ve múltiples ejemplos<br>✅ Benchmarking | ⚠️ No interactivo<br>⚠️ Usuarios fijos |

## 🎯 Conclusión

El sistema interactivo es perfecto para:
- ✅ Nuevos usuarios sin historial
- ✅ Exploración de géneros
- ✅ Descubrimiento de películas
- ✅ Uso casual y rápido

¡Disfruta descubriendo nuevas películas! 🍿
