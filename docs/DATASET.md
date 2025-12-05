# Dataset MovieLens 32M

Este directorio debe contener el dataset MovieLens 32M.

## Descarga

El dataset NO está incluido en el repositorio debido a su tamaño (~900MB).

### Opción 1: Descarga Manual

1. Visita: https://grouplens.org/datasets/movielens/32m/
2. Descarga el archivo `ml-32m.zip`
3. Extrae los archivos en este directorio (`ml-32m/`)

### Opción 2: Descarga con Script (Linux/Mac)

```bash
cd ml-32m
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip
mv ml-32m/* .
rmdir ml-32m
rm ml-32m.zip
```

### Opción 3: Descarga con PowerShell (Windows)

```powershell
cd ml-32m
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-32m.zip" -OutFile "ml-32m.zip"
Expand-Archive -Path "ml-32m.zip" -DestinationPath "."
Move-Item -Path "ml-32m\*" -Destination "." -Force
Remove-Item -Path "ml-32m" -Recurse
Remove-Item -Path "ml-32m.zip"
```

## Archivos Esperados

Después de la descarga, deberías tener:

```
ml-32m/
├── README.txt          (incluido en el repo)
├── checksums.txt       (incluido en el repo)
├── movies.csv          (incluido en el repo - 4.2 MB)
├── ratings.csv         (DESCARGAR - 877 MB)
├── tags.csv            (DESCARGAR - 72 MB)
└── links.csv           (DESCARGAR - 1.9 MB)
```

## Verificación

Para verificar que los archivos se descargaron correctamente:

```bash
# Linux/Mac
md5sum *; cat checksums.txt

# Windows (requiere herramienta MD5)
# Compara los checksums manualmente con checksums.txt
```

## Contenido del Dataset

- **ratings.csv**: 32,000,204 valoraciones de películas
- **movies.csv**: 87,585 películas con títulos y géneros
- **tags.csv**: 2,000,072 tags aplicados por usuarios
- **links.csv**: Enlaces a IMDb y TMDb

## Licencia

Este dataset está sujeto a la licencia de uso de GroupLens Research.
Ver README.txt para más detalles.
