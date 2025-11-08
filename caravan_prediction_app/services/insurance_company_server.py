import logging
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


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

    def load_pipeline(self, pipeline_path: str) -> None:
        """
        Carga el pipeline desde archivo pickle.

        Args:
            pipeline_path: Ruta al archivo .pkl del pipeline
        """
        if not self._is_loaded:
            try:
                pipeline_path_obj = Path(pipeline_path)

                if not pipeline_path_obj.exists():
                    raise FileNotFoundError(f"Archivo de pipeline no encontrado: {pipeline_path}")

                logger.info(f"Cargando pipeline desde: {pipeline_path}")

                with open(pipeline_path_obj, 'rb') as f:
                    self._pipeline = joblib.load(pipeline_path_obj)

                self._pipeline_path = pipeline_path
                self._is_loaded = True

                logger.info("Pipeline cargado exitosamente en memoria")

            except Exception as e:
                logger.error(f"Error cargando pipeline: {e}")
                raise

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



