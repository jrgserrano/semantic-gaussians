# Semantic Gaussians

Este repositorio contiene el código desarrollado en nuestro proyecto, el cual se ha utilizado como base el código de **[OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d)**.

## Instalación

```bash
git clone https://github.com/jrgserrano/semantic-gaussians.git --recursive
```

Si usas `just`:

```bash
just setup
```

O si no:

```bash
uv sync
uv sync --extra compile
```

## Checkpoints

Para crear la carpeta de checkpoints y descargar los checkpoints necesarios para usar los modelos fundacionales en el entrenamiento, ejecuta el siguiente comando:

```bash
just download_ckpts
```

Si no deseas usar `just`, puedes ejecutar los comandos correspondientes del `justfile` manualmente.

## Preparación de los Datasets

Para este proyecto utilizamos el dataset **Replica**, disponible a través de la página de la ETH Zurich. Puedes instalarlo siguiendo estos pasos:

```bash
mkdir -p datasets/Replica
cd datasets/Replica

wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
```

*Asegúrate de ajustar las rutas en los comandos posteriores según dónde hayas descomprimido los datos.*

## Generación de Nubes de Puntos Densas

Si deseas entrenar usando nuestra inicialización densa, puedes interceptar y generar un archivo `.ply` inicial (Dense Point Cloud). Por ejemplo, para crear una nube base de 1.000.000 de puntos para la escena `office4`:

```bash
uv run python -m core.data.preprocessing.extract_replica_pcd \
    /ruta/a/tus/datasets/Replica/office4 \
    --max_points 1000000
```
*Esto generará el archivo `points3d.ply` dentro del directorio de la escena.*

## Entrenamiento Completo

Una vez tienes la nube de puntos lista, puedes lanzar el entrenamiento completo. Este proceso entrena la geometría e incluye la generación de categorías y descripciones semánticas (el módulo de lenguaje `lang.enabled` está activado por defecto en la configuración).

```bash
uv run python core/train.py --config configs/replica.yaml \
    model.source_path=/ruta/a/tus/datasets/Replica/office4 \
    model.model_path=outputs/Replica/office4_semantic_full \
    model.init_type=ply \
    model.init_ply=/ruta/a/tus/datasets/Replica/office4/points3d.ply \
    --no-server
```

## Generación de Bounding Boxes (BBoxes) y JSON

Una vez finalizado el entrenamiento y generados los clusters de la escena, puedes calcular las bounding boxes (cajas delimitadoras) y exportar todas las instancias junto con sus descripciones a un formato `.json`.

1. **Generar las Bounding Boxes (`bboxes.pth`):**
```bash
uv run python core/semantic/generate_bboxes.py outputs/Replica/office4_semantic_full
```

2. **Exportar a JSON (`instances.json`):**
```bash
uv run python core/semantic/export_json.py outputs/Replica/office4_semantic_full
```

## Visualización de la Escena

Para lanzar el visualizador interactivo de la escena entrenada, puedes ejecutar el siguiente comando:

```bash
uv run python -m core.visualizer --model_dir outputs/Replica/office4_semantic_full --port 8080
```

O si prefieres utilizar `just`:

```bash
just visualize outputs/Replica/office4_semantic_full
```

Una vez ejecutado, abre tu navegador web y accede a `http://localhost:8080` para interactuar con la reconstrucción y la información semántica generada.

## Licencia

El código es distribuido bajo la licencia Gaussian-Splatting License. Para más información, visita el archivo LICENSE.
