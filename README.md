MNA_MLOps_Equipo12
==============================

Project created for the MLOps assignature in MNA-V Tec de Monterrey

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Aplicación de Predicción de Seguros de Caravanas

Este README explica los pasos necesarios para configurar, ejecutar el pipeline de datos, y desplegar la aplicación usando Docker.

--------------------------------------------------------------------------------

1. Ejecución del Pipeline de Procesamiento de Datos

El pipeline se encarga de leer los datos crudos, ejecutar todo el procesamiento estructurado y guardar el modelo necesario en la ubicación requerida por la aplicación.

Instrucciones

Para preparar el modelo y dejarlo en la ruta correcta (src/pipeline), ejecuta el siguiente archivo:

python src/pipelines/main_pipeline.py

* Función: Este script lee la data raw (df original), ejecuta toda la programación estructurada del flujo de trabajo de Machine Learning, y finaliza guardando el archivo del pipeline en la carpeta src.
* Importante: El archivo del pipeline debe existir en la ruta src para que la aplicación pueda levantarse y cargarlo correctamente para realizar las predicciones.

--------------------------------------------------------------------------------

2. Despliegue de la Aplicación con Docker

Para desplegar la aplicación como un servicio, usaremos Docker.

Requisitos

* Tener Docker instalado en tu sistema.

Pasos para el Despliegue

A. Crear el archivo de Variables de Entorno (.env)

Antes de construir la imagen, asegúrate de crear y configurar un archivo llamado .env en la raíz del proyecto. Este archivo debe contener, como mínimo, la siguiente configuración:

API_WORKERS=1

B. Construir la Imagen de Docker

Ejecuta el siguiente comando en la terminal desde la raíz del proyecto para construir la imagen.

sudo docker build -t caravan_predictions .

* Etiqueta: La imagen se etiquetará como caravan_predictions.

C. Ejecutar el Contenedor de Docker

Una vez que la imagen esté construida, ejecuta el siguiente comando para levantar la aplicación.

docker run -p 8080:8080 --env-file .env caravan_predictions

* Mapeo de Puertos: Mapea el puerto 8080 del host al puerto 8080 del contenedor.
* Variables de Entorno: Carga las variables definidas en el archivo .env.

--------------------------------------------------------------------------------

# 🚀 Instrucciones para desplegar **caravan_prediction:v1.x.x**

## 🐳 1. Descargar desde Docker Hub (si está disponible online)
```bash
docker pull caravan_prediction:v1.x.x
docker run -p 8080:8080 --env-file .env caravan_prediction:v1.x.x
```

> 💡 *Asegúrate de reemplazar `v1.x.x` por la versión más reciente publicada.*

---

## 🛠️ 2. Construir la imagen desde el repositorio local
Si prefieres compilar la imagen tú mismo, utiliza los siguientes comandos:

```bash
# Construir la imagen
docker build -t caravan_prediction:v1.0.0 .

# Ejecutar la aplicación en el puerto 8080 usando el archivo .env
docker run -p 8080:8080 --env-file .env caravan_prediction:v1.0.0
```

> ⚙️ Esto levantará el servicio FastAPI en http://localhost:8080

---

## 🔐 3. Archivo `.env` — Variables necesarias

Tu archivo `.env` debe incluir las siguientes variables de entorno, esenciales para la conexión con AWS S3 y la configuración del API:

| Variable | Descripción |
|-----------|--------------|
| **AWS_DEFAULT_REGION** | Región de AWS donde se aloja el bucket (ejemplo: `us-east-2`). |
| **AWS_ACCESS_KEY_ID** | ID de la clave de acceso para autenticación en AWS. |
| **AWS_SECRET_ACCESS_KEY** | Clave secreta asociada al `AWS_ACCESS_KEY_ID`. |
| **S3_BUCKET_NAME** | Nombre del bucket en S3 donde se almacenan los modelos y pipelines (por defecto `mna-tec-mlops`). |
| **S3_MODEL_KEY** | Ruta completa del modelo dentro del bucket (por ejemplo `h2o_models/models/GBM_3_AutoML_1_20251112_191136`). |
| **API_WORKERS** | Número de workers de Uvicorn que manejarán las peticiones del API (por defecto `1`). |
| **BEST_THRESHOLD** | Umbral de decisión para el clasificador H2O (por defecto `0.1`). |

---

## 🧩 4. Ejemplo de archivo `.env`
```env
AWS_DEFAULT_REGION=us-east-2
AWS_ACCESS_KEY_ID=TU_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=TU_SECRET_KEY
S3_BUCKET_NAME=mna-tec-mlops
S3_MODEL_KEY=h2o_models/models/GBM_3_AutoML_1_20251112_191136
API_WORKERS=1
BEST_THRESHOLD=0.1
S3_MODEL_PATH=h2o_models/models
```

---

## ✅ 5. Verificación del despliegue
Una vez el contenedor esté en ejecución, abre tu navegador o usa `curl`:

```bash
curl http://localhost:8080/docs
```

Esto debería mostrar la documentación interactiva del API (**Swagger UI**) para el servicio de predicción `caravan_prediction`.

--------------------------------------------------------------------------------

3. Uso del Endpoint de Predicción

Una vez que la aplicación esté levantada y el contenedor de Docker esté ejecutándose, puedes utilizar el siguiente endpoint para enviar un archivo CSV y recibir las predicciones.

Endpoint de Predicción

Utiliza el siguiente comando curl para interactuar con la API:

curl --location 'http://localhost:8080/api/caravan-prediction/predict' \
--form 'file=@"{local_path}/MNA_MLOps_Equipo12_backup/tests/data_tests/prueba_mlops.csv"'

* Reemplazar {local_path}: Asegúrate de sustituir {local_path} con la ruta absoluta de tu máquina donde se encuentra el archivo de prueba (e.g., /home/user/my_project).
* Método: POST (implícito con --form).
* Formato de Datos: El endpoint espera un archivo CSV subido a través de un form-data con la clave file.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
