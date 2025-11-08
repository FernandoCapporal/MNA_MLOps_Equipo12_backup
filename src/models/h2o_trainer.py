import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import logging
from typing import Union, Optional, Dict, Any
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

logger = logging.getLogger('preprocessor_logger')
logger.setLevel(logging.INFO)


class H2OAutoMLTrainer:
    def __init__(self, max_models: int = 20, max_runtime_secs: int = 300,
                 seed: int = 42, balance_classes: bool = True,
                 stopping_metric: str = "AUC"):
        """
        Inicializa el entrenador de H2O AutoML.

        Args:
            max_models: Número máximo de modelos a entrenar
            max_runtime_secs: Tiempo máximo de ejecución en segundos
            seed: Semilla para reproducibilidad
            balance_classes: Balancear clases para datos desbalanceados
            stopping_metric: Métrica para early stopping
        """
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.balance_classes = balance_classes
        self.stopping_metric = stopping_metric

        # Atributos del modelo
        self.aml = None
        self.leader = None
        self.leaderboard = None
        self.h2o_frame = None

        # Estado de H2O
        self.h2o_initialized = False

        logger.info(f"H2OAutoMLTrainer inicializado - max_models: {max_models}, "
                    f"max_runtime_secs: {max_runtime_secs}")

    def initialize_h2o(self) -> None:
        """Inicializa la conexión con H2O."""
        if not self.h2o_initialized:
            try:
                h2o.init()
                self.h2o_initialized = True
                logger.info("H2O inicializado exitosamente")
            except Exception as e:
                logger.error(f"Error inicializando H2O: {e}")
                raise

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target') -> None:
        """
        Prepara los datos para H2O AutoML.

        Args:
            df: DataFrame con los datos
            target_col: Nombre de la columna target
        """
        logger.info(f"Preparando datos - Shape: {df.shape}, Target: {target_col}")

        if not self.h2o_initialized:
            self.initialize_h2o()

        # Convertir a H2OFrame
        self.h2o_frame = h2o.H2OFrame(df)

        # Convertir target a factor (para clasificación)
        if target_col in self.h2o_frame.columns:
            self.h2o_frame[target_col] = self.h2o_frame[target_col].asfactor()
            logger.info(f"Target '{target_col}' convertido a factor")

        logger.info(f"Datos preparados - H2OFrame shape: {self.h2o_frame.shape}")

    def train(self, target_col: str = 'target',
              training_frame: Optional[h2o.H2OFrame] = None) -> None:
        """
        Entrena el modelo AutoML.

        Args:
            target_col: Columna target
            training_frame: Frame de entrenamiento (opcional, usa self.h2o_frame por defecto)
        """
        logger.info("Iniciando entrenamiento AutoML")

        if training_frame is None:
            if self.h2o_frame is None:
                raise ValueError("No hay datos preparados. Ejecute prepare_data primero.")
            training_frame = self.h2o_frame

        # Configurar AutoML
        self.aml = H2OAutoML(
            max_models=self.max_models,
            seed=self.seed,
            max_runtime_secs=self.max_runtime_secs,
            balance_classes=self.balance_classes,
            stopping_metric=self.stopping_metric
        )

        # Entrenar
        self.aml.train(y=target_col, training_frame=training_frame)

        # Obtener resultados
        self.leader = self.aml.leader
        self.leaderboard = self.aml.leaderboard

        logger.info("Entrenamiento AutoML completado exitosamente")
        self._log_training_summary()

    def _log_training_summary(self) -> None:
        """Registra un resumen del entrenamiento."""
        if self.leaderboard is not None:
            lb_head = self.leaderboard.head()
            logger.info("Leaderboard - Top modelos:")
            for i, (model_id, *metrics) in enumerate(lb_head.as_data_frame().itertuples(index=False)):
                logger.info(f"  {i + 1}. {model_id} - {metrics}")

        if self.leader is not None:
            logger.info(f"Modelo líder: {self.leader.model_id}")

    def predict(self, X_data: Union[pd.DataFrame, h2o.H2OFrame],
                best_threshold: Optional[float] = None,
                auto_threshold: bool = False,
                y_true: Optional[np.ndarray] = None,
                return_h2o_frame: bool = False) -> Union[pd.DataFrame, h2o.H2OFrame]:
        """
        Realiza predicciones con el modelo líder.

        Args:
            X_data: Datos para predecir
            best_threshold: Umbral óptimo para clasificación (opcional)
            auto_threshold: Si calcular automáticamente el threshold que maximiza AUC
            y_true: Valores reales (requerido si auto_threshold=True)
            return_h2o_frame: Si retornar H2OFrame en lugar de DataFrame

        Returns:
            DataFrame o H2OFrame con predicciones
        """
        logger.info("Realizando predicciones")

        if self.leader is None:
            raise ValueError("No hay modelo entrenado. Ejecute train primero.")

        # Validar parámetros para auto_threshold
        if auto_threshold and y_true is None:
            raise ValueError("y_true es requerido cuando auto_threshold=True")

        # Convertir a H2OFrame si es necesario
        if isinstance(X_data, pd.DataFrame):
            h2o_X = h2o.H2OFrame(X_data)
            logger.info(f"DataFrame convertido a H2OFrame - Shape: {h2o_X.shape}")
        else:
            h2o_X = X_data

        # Realizar predicciones
        predictions = self.leader.predict(h2o_X)

        if return_h2o_frame:
            # Combinar features con predicciones
            results_h2o = h2o_X.cbind(predictions)
            logger.info("Predicciones completadas - Retornando H2OFrame")
            return results_h2o
        else:
            # Convertir a DataFrame de pandas
            features_df = h2o_X.as_data_frame().copy()
            probabilities = predictions['p1'].as_data_frame().values.flatten()
            predicted_classes = predictions['predict'].as_data_frame().values.flatten()

            results_df = features_df.copy()
            results_df['predicted_prob'] = probabilities
            results_df['predicted_class'] = predicted_classes

            # Calcular threshold automáticamente si se solicita
            if auto_threshold:
                # Calcular curva ROC
                fpr, tpr, thresholds = roc_curve(y_true, probabilities)
                roc_auc = auc(fpr, tpr)

                # Encontrar threshold que maximiza AUC (punto más alejado de la diagonal)
                optimal_idx = np.argmax(tpr - fpr)
                best_threshold = thresholds[optimal_idx]

                logger.info(f"Threshold automático calculado: {best_threshold:.4f} (AUC: {roc_auc:.4f})")

            # Aplicar threshold si se proporciona o se calcula automáticamente
            if best_threshold is not None:
                logger.info(f"Umbral aplicado: {best_threshold}")

                results_df['prediction'] = ((results_df['predicted_prob'] >= best_threshold) &
                                            (results_df['predicted_class'] == 1)).astype(int)
            else:
                # Si no hay threshold, usar predicted_class directamente
                results_df['prediction'] = results_df['predicted_class']

            logger.info(f"Predicciones completadas - DataFrame shape: {results_df.shape}")
            return results_df

    def predict_with_actuals(self, X_data: Union[pd.DataFrame, h2o.H2OFrame],
                             y_actual: pd.Series,
                             best_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Realiza predicciones e incluye los valores reales.

        Args:
            X_data: Features para predecir
            y_actual: Valores reales del target
            best_threshold: Umbral óptimo para clasificación

        Returns:
            DataFrame con predicciones y valores reales
        """
        logger.info("Realizando predicciones con valores reales")

        # Obtener predicciones base
        results_df = self.predict(X_data, best_threshold=best_threshold)

        # Agregar valores reales
        results_df['actual'] = y_actual.values

        logger.info(f"Predicciones con valores reales completadas - Shape: {results_df.shape}")
        return results_df

    def get_model_performance(self, test_data: Optional[h2o.H2OFrame] = None,
                              target_col: str = 'target') -> Any:
        """
        Evalúa el performance del modelo líder.

        Args:
            test_data: Datos de test (opcional)
            target_col: Columna target

        Returns:
            Objeto de performance del modelo
        """
        logger.info("Evaluando performance del modelo")

        if self.leader is None:
            raise ValueError("No hay modelo entrenado. Ejecute train primero.")

        if test_data is None:
            if self.h2o_frame is None:
                raise ValueError("No hay datos disponibles para evaluación")
            test_data = self.h2o_frame

        performance = self.leader.model_performance(test_data=test_data)

        # Log de métricas principales
        if hasattr(performance, 'auc'):
            logger.info(f"AUC: {performance.auc():.4f}")
        if hasattr(performance, 'logloss'):
            logger.info(f"LogLoss: {performance.logloss():.4f}")
        if hasattr(performance, 'accuracy'):
            logger.info(f"Accuracy: {performance.accuracy():.4f}")

        return performance

    def get_leaderboard(self, as_dataframe: bool = True) -> Union[h2o.H2OFrame, pd.DataFrame]:
        """
        Obtiene el leaderboard de modelos.

        Args:
            as_dataframe: Si retornar como DataFrame de pandas

        Returns:
            Leaderboard de modelos
        """
        if self.leaderboard is None:
            raise ValueError("No hay leaderboard disponible. Ejecute train primero.")

        if as_dataframe:
            return self.leaderboard.as_data_frame()
        else:
            return self.leaderboard

    def save_model(self, model_name: str = "h2o_automl_model", base_path: str = "./") -> str:
        """
        Guarda el modelo en disco con nombre configurable.

        Args:
            model_name: Nombre del modelo (sin extensión)
            base_path: Ruta base donde guardar el modelo

        Returns:
            Ruta completa donde se guardó el modelo
        """
        if self.leader is None:
            raise ValueError("No hay modelo líder para guardar")

        # Crear directorio si no existe
        os.makedirs(base_path, exist_ok=True)

        # Guardar modelo
        model_path = h2o.save_model(
            model=self.leader,
            path=base_path,
            force=True,
            filename=model_name
        )

        logger.info(f"Modelo guardado en: {model_path}")

    def shutdown_h2o(self) -> None:
        """Cierra la conexión con H2O."""
        if self.h2o_initialized:
            h2o.cluster().shutdown()
            self.h2o_initialized = False
            logger.info("H2O shutdown completado")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen del entrenamiento.

        Returns:
            Diccionario con información del entrenamiento
        """
        summary = {
            'max_models': self.max_models,
            'max_runtime_secs': self.max_runtime_secs,
            'seed': self.seed,
            'balance_classes': self.balance_classes,
            'stopping_metric': self.stopping_metric,
            'leader_model': self.leader.model_id if self.leader else None,
            'models_trained': len(self.leaderboard) if self.leaderboard else 0,
            'h2o_initialized': self.h2o_initialized
        }

        if self.leaderboard is not None and len(self.leaderboard) > 0:
            lb_df = self.leaderboard.as_data_frame()
            summary['top_model_auc'] = lb_df.iloc[0]['auc'] if 'auc' in lb_df.columns else None

        return summary

    def __enter__(self):
        """Support for context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context manager exit."""
        self.shutdown_h2o()
