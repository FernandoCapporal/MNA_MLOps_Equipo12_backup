import logging
import joblib
import numpy as np
from typing import Optional, Dict, Any
import pandas as pd
import tempfile
from caravan_prediction_app.application.settings import Settings

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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineSingleton, cls).__new__(cls)
        return cls._instance

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

            self._pipeline_path = latest_pipeline_path
            self._is_loaded = True
            logger.info(f"Pipeline cargado exitosamente desde {latest_pipeline_path}")

        except Exception as e:
            logger.warning(f"No se pudo cargar el pipeline desde S3: {e}")
            self._load_dummy_pipeline()

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
        logger.info("Realizando predicciones con pipeline cargado...")
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


import boto3
import logging

logger = logging.getLogger(__name__)


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
