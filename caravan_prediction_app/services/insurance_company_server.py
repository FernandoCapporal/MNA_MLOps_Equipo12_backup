import joblib
import numpy as np
from typing import Optional, Dict, Any
import pandas as pd
import tempfile
import boto3
import logging
import os
import h2o
from caravan_prediction_app.application.settings import Settings
from fastapi import HTTPException
from io import BytesIO

logger = logging.getLogger(__name__)
settings = Settings()


class PipelineSingleton:
    """
    Singleton para cargar y mantener pipeline desde archivo pickle.
    """
    _instance = None
    _pipeline = None
    _is_loaded = False
    _pipeline_path = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineSingleton, cls).__new__(cls)
        return cls._instance

    def load_model(self, s3_key: str = None) -> None:
        """
        Descarga y carga un modelo H2O desde S3.
        """
        try:
            s3_model_path = s3_key or os.getenv("S3_MODEL_PATH")
            if not s3_model_path:
                raise ValueError("Configura S3_MODEL_PATH o pasa s3_key")

            logger.info(f"Cargando modelo H2O desde: {s3_model_path}")

            # Inicializar H2O primero
            try:
                h2o.init()
                logger.info("H2O inicializado")
            except Exception as e:
                logger.warning(f"H2O ya inicializado o error: {e}")

            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_DEFAULT_REGION,
            )

            bucket = settings.S3_BUCKET_NAME
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_model_path = os.path.join(current_dir, "models")

            full_path = self._find_latest_model(s3, bucket, s3_model_path)
            splited = full_path.split("/")
            if len(splited) > 1:
                logger.info(f"El nombre del último modelo encontrado es {splited[-1]}")
                self._model_name = splited[-1]
            else:
                self._model_name = full_path

            logger.info(f"Subiendo modelo H2O a: {local_model_path}")

            # Descargar si no existe
            if not os.path.exists(local_model_path):
                logger.info("Descargando modelo desde S3...")
                os.makedirs(local_model_path, exist_ok=True)

                # Usar un enfoque más seguro para la descarga
                self._safe_download_model(s3, bucket, s3_model_path, local_model_path)
            else:
                logger.info("Modelo ya existe localmente")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def _find_latest_model(self, s3_client, bucket: str, s3_folder_path: str) -> str:
        """
        Busca el objeto más reciente (último modelo) dentro de una carpeta en S3.

        Args:
            s3_client: Cliente boto3 de S3 ya inicializado
            bucket (str): Nombre del bucket
            s3_folder_path (str): Prefijo o ruta tipo 'models/h2o_models/'

        Returns:
            str: Nombre (key) del último objeto encontrado en esa carpeta
        """
        import botocore

        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            latest_obj = None
            latest_date = None

            for page in paginator.paginate(Bucket=bucket, Prefix=s3_folder_path):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue

                    last_modified = obj['LastModified']

                    if latest_date is None or last_modified > latest_date:
                        latest_date = last_modified
                        latest_obj = key

            if not latest_obj:
                raise FileNotFoundError(f"No se encontraron modelos en s3://{bucket}/{s3_folder_path}")

            logger.info(f"Último modelo encontrado: {latest_obj} (modificado: {latest_date})")
            return latest_obj

        except botocore.exceptions.ClientError as e:
            logger.error(f"Error accediendo a S3: {e}")
            raise

    def _safe_download_model(self, s3_client, bucket, s3_prefix, local_path):
        """
        Descarga el modelo de forma segura evitando archivos temporales del sistema.
        """
        try:
            # Listar objetos de forma paginada
            paginator = s3_client.get_paginator('list_objects_v2')
            downloaded_files = 0

            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    s3_key = obj['Key']

                    # Saltar directorios y archivos temporales del sistema
                    if s3_key.endswith('/') or '/.' in s3_key:
                        continue

                    # Calcular ruta local
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_file_path = os.path.join(local_path, relative_path)

                    # Crear directorio padre si no existe
                    # os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    try:
                        # Descargar archivo directamente
                        s3_client.download_file(bucket, s3_key, local_file_path)
                        downloaded_files += 1
                        logger.debug(f"Descargado: {s3_key}")
                    except Exception as file_error:
                        logger.warning(f"Error descargando {s3_key}: {file_error}")
                        continue

            logger.info(f"Descarga completada. Archivos descargados: {downloaded_files}")

            if downloaded_files == 0:
                raise ValueError("No se pudieron descargar archivos del modelo")

        except Exception as e:
            logger.error(f"Error en descarga segura: {e}")
            # Limpiar en caso de error
            if os.path.exists(local_path):
                import shutil
                shutil.rmtree(local_path)
            raise

    def load_pipeline(self, folder_name: str) -> None:
        try:
            logger.info(f"Intentando cargar pipeline desde S3")

            latest_pipeline_path = get_latest_pipeline_from_s3(folder_name)

            bucket = settings.S3_BUCKET_NAME
            key = "/".join(latest_pipeline_path.replace(f"s3://{bucket}/", "").split("/"))

            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_DEFAULT_REGION,
            )

            with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
                s3.download_file(bucket, key, tmp_file.name)
                tmp_file.flush()
                self._pipeline = joblib.load(tmp_file.name)

            self._connect_h2o_model()

            self._pipeline_path = latest_pipeline_path
            self._is_loaded = True
            logger.info(f"Pipeline cargado exitosamente desde {latest_pipeline_path}")

        except Exception as e:
            logger.warning(f"No se pudo cargar el pipeline desde S3: {e}")
            self._load_dummy_pipeline()

    def _connect_h2o_model(self):
        """
        Conecta el pipeline cargado con el modelo H2O.
        Asume que el modelo H2O está disponible localmente en la ruta esperada.
        """
        try:
            # Calcular ruta local del modelo
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_model_path = os.path.join(current_dir, "models", self._model_name)

            logger.info(f"Conectando con modelo H2O en: {local_model_path}")

            # Inicializar H2O si no está inicializado
            try:
                h2o.init()
                logger.info("H2O inicializado")
            except Exception as e:
                logger.warning(f"H2O ya inicializado o error: {e}")

            # Cargar modelo H2O
            if os.path.exists(local_model_path):
                self._h2o_model = h2o.load_model(local_model_path)
                logger.info(f"Modelo H2O cargado:")
            else:
                logger.warning(f"Modelo H2O no encontrado en {local_model_path}. Se necesitará descargar.")

            self._create_extended_pipeline()

        except Exception as e:
            logger.error(f"Error conectando modelo H2O: {e}")
            raise

    def _create_extended_pipeline(self):
        """
        Crea un pipeline extendido que combina el preprocesamiento con la predicción H2O.
        """

        class ExtendedPipeline:
            def __init__(self, preprocessing_pipeline, h2o_model, best_threshold=None):
                self.preprocessing_pipeline = preprocessing_pipeline
                self.h2o_model = h2o_model
                self.best_threshold = best_threshold
                self._final_estimator = h2o_model  # Para compatibilidad

            def fit(self, X, y=None):
                # No necesitamos fit para inferencia
                return self

            def transform(self, X):
                # Solo preprocesamiento
                return self.preprocessing_pipeline.fit_transform(X)

            def predict(self, X):
                X_processed = self.preprocessing_pipeline.fit_transform(X)
                h2o_df = h2o.H2OFrame(X_processed)
                predictions = self.h2o_model.predict(h2o_df)

                # Extraer probabilidades y clases como antes
                probabilities = predictions['p1'].as_data_frame().values.flatten()
                predicted_classes = predictions['predict'].as_data_frame().values.flatten()

                if self.best_threshold is not None:
                    final_predictions = (probabilities >= self.best_threshold).astype(int)
                    logger.info(f"Umbral aplicado: {self.best_threshold}")
                else:
                    final_predictions = predicted_classes

                return final_predictions

            def fit_transform(self, X, y=None):
                # Para compatibilidad con tu código existente
                predictions = self.predict(X)
                return pd.DataFrame({'prediction': predictions})

        # Obtener best_threshold del pipeline original si existe
        best_threshold = getattr(self._pipeline, 'best_threshold', None)
        logger.info(f"Best threshold: {best_threshold}")
        if best_threshold is None:
            best_threshold = settings.BEST_THRESHOLD
            logger.info(f"Best threshold updated: {best_threshold}")

        # Crear pipeline extendido
        self._pipeline = ExtendedPipeline(
            preprocessing_pipeline=self._pipeline,
            h2o_model=self._h2o_model,
            best_threshold=best_threshold
        )

    def _load_dummy_pipeline(self) -> None:
        """ Clasificador aleatorio temporal con metodo fit_transform."""

        class DummyPipeline:
            def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
                n = len(X)
                preds = np.random.choice([0, 1], size=n)
                return pd.DataFrame({"prediction": preds})

        self._pipeline = DummyPipeline()
        self._is_loaded = True
        self._pipeline_path = "dummy_in_memory"
        logger.info("Clasificador aleatorio cargado en memoria (dummy fallback)")

    def get_pipeline(self):
        """
        Retorna el pipeline cargado.

        Returns:
            Pipeline de scikit-learn
        """
        if not self._is_loaded or self._pipeline is None:
            raise RuntimeError("Pipeline no cargado. Llama a load_pipeline() primero.")
        return self._pipeline

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones usando el pipeline cargado.

        Args:
            X: DataFrame con features para predecir

        Returns:
            DataFrame con predicciones
        """
        pipeline = self.get_pipeline()
        logger.info(f"Realizando predicciones con pipeline cargado... {X.columns}")
        results = pipeline.fit_transform(X)
        X['prediction'] = results.values
        return X

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna el estado del singleton.

        Returns:
            Dict con información del estado
        """
        return {
            'is_loaded': self._is_loaded,
            'pipeline_loaded': self._pipeline is not None,
            'pipeline_path': self._pipeline_path
        }

    def reload_pipeline(self, pipeline_path: Optional[str] = None) -> None:
        """
        Recarga el pipeline (útil para actualizaciones).

        Args:
            pipeline_path: Nueva ruta (opcional, usa la misma si no se especifica)
        """
        if pipeline_path is None:
            if self._pipeline_path is None:
                raise ValueError("No hay ruta de pipeline para recargar")
            pipeline_path = self._pipeline_path

        self._pipeline = None
        self._is_loaded = False
        self.load_pipeline(pipeline_path)

    def reset(self) -> None:
        """
        Reinicia el singleton.
        """
        self._pipeline = None
        self._is_loaded = False
        self._pipeline_path = None
        logger.info("Singleton reiniciado")


def get_latest_pipeline_from_s3(folder_name: str) -> str:
    bucket = settings.S3_BUCKET_NAME
    prefix = folder_name.rstrip("/") + "/"

    logger.info(f"Buscando archivo .pkl más reciente en s3://{bucket}/{prefix}")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION,
    )

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        logger.error(f"No se encontraron objetos en s3://{bucket}/{prefix}")
        raise FileNotFoundError(f"No se encontraron objetos en s3://{bucket}/{prefix}")

    pkl_objects = [obj for obj in response["Contents"] if obj["Key"].endswith(".pkl")]

    if not pkl_objects:
        logger.error(f"No se encontraron archivos .pkl en s3://{bucket}/{prefix}")
        raise FileNotFoundError(f"No se encontraron archivos .pkl en s3://{bucket}/{prefix}")

    pkl_objects.sort(key=lambda x: x["LastModified"], reverse=True)

    latest_obj = pkl_objects[0]
    latest_key = latest_obj["Key"]
    latest_path = f"s3://{bucket}/{latest_key}"

    logger.info(f"Último pipeline encontrado: {latest_path} ({latest_obj['LastModified']})")

    return latest_path


def load_and_format_dataframe(contents: bytes) -> pd.DataFrame:
    """
    Carga un CSV desde bytes, asigna nombres de columnas esperados
    y agrega la columna 'target' si el archivo tiene solo 85 columnas.

    Args:
        contents (bytes): Contenido del archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con columnas formateadas correctamente.
    """
    logger.info(f"Cargando archivo load_and_format_dataframe...")
    sociodemographic_cols = [f"SD_{i}" for i in range(1, 44)]
    product_cols = [f"PD_{i - 44}" for i in range(44, 86)]
    cols = sociodemographic_cols + product_cols + ["target"]

    # Leer el CSV desde memoria
    df = pd.read_csv(BytesIO(contents), header=None, names=cols)

    # Si hay solo 85 columnas, agregar dummy 'target'
    if df.shape[1] == 85:
        df["target"] = 0  # valor dummy

    logger.info(f"DataFrame cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def load_and_format_json_dataframe(data: list[dict]) -> pd.DataFrame:
    """
    Convierte una lista de diccionarios JSON en un DataFrame con
    columnas formateadas correctamente y agrega 'target' si falta.

    Args:
        data (list[dict]): Lista de registros, cada uno con columnas col_1...col_85 (o col_86).

    Returns:
        pd.DataFrame: DataFrame listo para pasar al pipeline.
    """
    logger.info(f"Cargando archivo load_and_format_json_dataframe...")
    df = pd.DataFrame(data)
    logger.info(f"JSON parsed successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    if df.shape[1] not in [85, 86]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid number of columns: {df.shape[1]} (expected 85 or 86)."
        )

    sociodemographic_cols = [f"SD_{i}" for i in range(1, 44)]
    product_cols = [f"PD_{i - 44}" for i in range(44, 86)]
    expected_cols = sociodemographic_cols + product_cols + ["target"]

    # Si no hay target, agregar dummy
    if df.shape[1] == 85:
        df["target"] = 0
        logger.warning("JSON had 85 columns. Added dummy 'target' column.")

    df.columns = expected_cols
    logger.info(f"DataFrame ready: {df.shape[0]} rows, {df.shape[1]} columns.")

    return df
