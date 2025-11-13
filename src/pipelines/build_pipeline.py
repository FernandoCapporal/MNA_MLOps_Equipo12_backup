import joblib
from sklearn.base import BaseEstimator
import logging
import pandas as pd
import numpy as np
import h2o
from sklearn.pipeline import Pipeline
from src.data.make_dataset_and_model import run_insurance_pipeline
import os
import io
import boto3
from datetime import datetime

logger = logging.getLogger('preprocessor_logger')
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SociodemographicToZoneTransformer(BaseEstimator):
    def __init__(self, sociodemographic_cols, zone_mapper_df):
        """
        Transforma columnas sociodemográficas a zona y aplica one-hot encoding.

        Args:
            sociodemographic_cols: Lista de columnas sociodemográficas
            zone_mapper_df: DataFrame con mapeo sociodemográfico -> zona
        """
        self.sociodemographic_cols = sociodemographic_cols
        self.zone_mapper_df = zone_mapper_df
        self.available_zones_ = None

    def fit(self, X, y=None):
        logger.info("Fitteando SociodemographicToZoneTransformer")
        # Determinar las zonas disponibles del mapper
        self.available_zones_ = sorted(self.zone_mapper_df['zone'].unique())
        logger.info(f"Zonas disponibles: {self.available_zones_}")
        return self

    def transform(self, X):
        logger.info("Transformando columnas sociodemográficas a zona")
        X_transformed = X.copy()

        # 1. Mapear a zona
        X_transformed = X_transformed.merge(
            self.zone_mapper_df,
            on=self.sociodemographic_cols,
            how='left'
        )

        # Verificar zonas no mapeadas
        zonas_nulas = X_transformed['zone'].isna().sum()
        if zonas_nulas > 0:
            logger.warning(f"{zonas_nulas} registros sin zona asignada")
            # Rellenar con zona más frecuente o valor por defecto
            X_transformed['zone'] = X_transformed['zone'].fillna(self.available_zones_[0])

        # 2. Aplicar one-hot encoding
        zone_dummies = pd.get_dummies(X_transformed['zone'], prefix='zone')

        # Asegurar que tenemos todas las zonas esperadas
        for zone in self.available_zones_:
            col_name = f'zone_{zone}'
            if col_name not in zone_dummies.columns:
                zone_dummies[col_name] = 0

        # 3. Eliminar columnas originales
        columns_to_drop = self.sociodemographic_cols + ['zone']
        X_transformed = X_transformed.drop(columns=columns_to_drop)

        # 4. Concatenar con one-hot encoding
        X_transformed = pd.concat([X_transformed, zone_dummies], axis=1)

        logger.info(f"Transformación completada. Shape: {X_transformed.shape}")
        return X_transformed


class ColumnDropper(BaseEstimator):
    def __init__(self, columns_to_drop):
        """
        Elimina columnas especificadas del DataFrame.

        Args:
            columns_to_drop: Lista de columnas a eliminar
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        logger.info("Fitteando ColumnDropper")
        # Verificar qué columnas existen realmente
        self.existing_columns_to_drop_ = [
            col for col in self.columns_to_drop if col in X.columns
        ]
        logger.info(f"Columnas a eliminar: {self.existing_columns_to_drop_}")
        return self

    def transform(self, X):
        logger.info("Eliminando columnas especificadas")
        X_transformed = X.copy()

        if self.existing_columns_to_drop_:
            X_transformed = X_transformed.drop(columns=self.existing_columns_to_drop_)
            logger.info(f"Eliminadas {len(self.existing_columns_to_drop_)} columnas")

        logger.info(f"Transformación completada. Shape: {X_transformed.shape}")
        return X_transformed


class SkewnessCorrector(BaseEstimator):
    def __init__(self, power_params):
        """
        Aplica corrección de asimetría usando parámetros predefinidos.

        Args:
            power_params: Diccionario {columna: lambda} para transformación
        """
        self.power_params = power_params

    def fit(self, X, y=None):
        logger.info("Fitteando SkewnessCorrector")
        # Verificar qué columnas existen
        self.existing_power_params_ = {
            col: lambda_val for col, lambda_val in self.power_params.items()
            if col in X.columns
        }
        logger.info(f"Columnas para corrección: {list(self.existing_power_params_.keys())}")
        return self

    def transform(self, X):
        logger.info("Aplicando corrección de asimetría")
        X_transformed = X.copy()

        for col, lambda_val in self.existing_power_params_.items():
            # Aplicar transformación Yeo-Johnson
            if lambda_val == 0:
                # Log transformation
                X_transformed[col] = np.log1p(X_transformed[col])
            elif lambda_val == 1:
                # No transformation needed
                pass
            else:
                # General Yeo-Johnson transformation
                X_transformed[col] = ((X_transformed[col] + 1) ** lambda_val - 1) / lambda_val

            logger.debug(f"Transformada columna {col} con lambda: {lambda_val}")

        logger.info("Corrección de asimetría completada")
        return X_transformed


class H2OPredictor(BaseEstimator):
    def __init__(self, model_path, best_threshold=None):
        """
        Realiza predicciones usando modelo H2O.

        Args:
            model_path: Ruta al modelo H2O guardado
            best_threshold: Umbral para clasificación (opcional)
        """
        self.model_path = model_path
        self.best_threshold = best_threshold
        self.h2o_model = None
        self.h2o_initialized = False

    def _initialize_h2o(self):
        """Inicializa H2O si no está inicializado."""
        if not self.h2o_initialized:
            h2o.init()
            self.h2o_initialized = True
            logger.info("H2O inicializado")

    def fit(self, X, y=None):
        logger.info("Fitteando H2OPredictor")
        self._initialize_h2o()

        # Cargar modelo H2O
        self.h2o_model = h2o.load_model(self.model_path)
        logger.info(f"Modelo H2O cargado: {self.h2o_model.model_id}")

        return self

    def transform(self, X):
        logger.info("Realizando predicciones con H2O")
        X_transformed = X.copy()

        # Convertir a H2OFrame
        h2o_df = h2o.H2OFrame(X_transformed)

        # Realizar predicciones
        predictions = self.h2o_model.predict(h2o_df)

        # Extraer probabilidades y clases
        probabilities = predictions['p1'].as_data_frame().values.flatten()
        predicted_classes = predictions['predict'].as_data_frame().values.flatten()

        # Agregar al DataFrame resultante
        X_transformed['predicted_prob'] = probabilities
        X_transformed['predicted_class'] = predicted_classes

        # Aplicar threshold si se especifica
        if self.best_threshold is not None:
            X_transformed['prediction'] = (
                    X_transformed['predicted_prob'] >= self.best_threshold
            ).astype(int)
            logger.info(f"Umbral aplicado: {self.best_threshold}")

        logger.info(f"Predicciones completadas. Shape: {X_transformed.shape}")

        return X_transformed[['prediction']]

    def __del__(self):
        """Cleanup al destruir el objeto."""
        if self.h2o_initialized:
            h2o.cluster().shutdown()
            logger.info("H2O shutdown completado")


def create_prediction_pipeline(zone_mapper_df, sociodemographic_cols,
                               columns_to_drop, power_params,
                               h2o_model_path, best_threshold=None):
    """
    Crea un pipeline completo de scikit-learn para predicciones.

    Args:
        zone_mapper_df: DataFrame con mapeo sociodemográfico -> zona
        sociodemographic_cols: Columnas sociodemográficas
        columns_to_drop: Columnas a eliminar
        power_params: Parámetros para corrección de asimetría
        h2o_model_path: Ruta al modelo H2O
        best_threshold: Umbral para clasificación

    Returns:
        Pipeline de scikit-learn
    """
    pipeline = Pipeline([
        ('sociodemographic_mapper', SociodemographicToZoneTransformer(
            sociodemographic_cols=sociodemographic_cols,
            zone_mapper_df=zone_mapper_df
        )),
        ('column_dropper', ColumnDropper(
            columns_to_drop=columns_to_drop
        )),
        ('skewness_corrector', SkewnessCorrector(
            power_params=power_params
        )),
        ('h2o_predictor', H2OPredictor(
            model_path=h2o_model_path,
            best_threshold=best_threshold
        ))
    ])

    logger.info("Pipeline creado exitosamente")
    return pipeline


def main():
    logger.info("Iniciando creación automática del pipeline de predicción...")

    preprocessor, sociodemographic_cols = run_insurance_pipeline()

    """ Aqui se puede escoger el mejor modelo guardado para hacer inferencias directamente. """
    h2o_model_path = "../models/GBM_3_AutoML_1_20251112_191136/"

    prediction_pipeline = create_prediction_pipeline(
        zone_mapper_df=preprocessor.zone_mapper_df,
        sociodemographic_cols=sociodemographic_cols,
        columns_to_drop=preprocessor.cols_to_drop_,
        power_params=preprocessor.power_params_,
        h2o_model_path=h2o_model_path,
        best_threshold=0.1
    )

    # Variables de entorno para acceso a S3
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")  # valor por defecto

    if not all([aws_access_key, aws_secret_key]):
        logger.error("No se encontraron las credenciales de AWS en las variables de entorno.")
        return

    # Configuramos cliente de S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
    )

    # Nombre del bucket y clave del objeto
    bucket_name = os.getenv("S3_BUCKET_NAME", "")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    object_key = f"models/prediction_pipeline_{timestamp}.pkl"

    # Guardar el pipeline en un buffer de memoria
    buffer = io.BytesIO()
    joblib.dump(prediction_pipeline, buffer)
    buffer.seek(0)

    try:
        # Subir el archivo a S3
        s3.upload_fileobj(buffer, bucket_name, object_key)
        logger.info(f"Pipeline guardado exitosamente en S3: s3://{bucket_name}/{object_key}")
    except Exception as e:
        logger.error(f"Error al subir el pipeline a S3: {e}")


if __name__ == "__main__":
    main()
