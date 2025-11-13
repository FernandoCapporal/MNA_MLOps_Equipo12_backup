import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.h2o_trainer import H2OAutoMLTrainer
from src.features.feature_engineer import Preprocessor
import logging
from pathlib import Path
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


def run_insurance_pipeline():
    """
    Pipeline completo para procesamiento y entrenamiento de datos de seguros.

    Returns:
        tuple: (preprocessor, trainer, predictions, X_test, y_test)
    """
    # Configuración inicial
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / 'data' / 'raw' / 'insurance_company_original.csv'
    file_name = str(file_path)
    sociodemographic_cols = [f"SD_{i}" for i in range(1, 44)]
    product_cols = [f"PD_{i - 44}" for i in range(44, 86)]
    cols = sociodemographic_cols + product_cols + ["target"]

    # Carga y preparación inicial de datos
    df = pd.read_csv(file_name, header=None, names=cols)
    df = df.iloc[1:].reset_index(drop=True)  # Eliminar primera fila si es header

    # Preprocesamiento
    preprocessor = Preprocessor(df)
    processed = preprocessor.apply_preprocess(sociodemographic_cols, product_cols)

    """ El siguiente flujo se encuentra comentado para evitar el entrenamiento, ya que 
    desde los notebooks de experimentación ahora se generan los modelos y se guardan en la carpeta models.
    El user puede pasar como argumento el path del mejor modelo guardado para hacer inferencias directamente. """

    # Entrenamiento del modelo
    # trainer = H2OAutoMLTrainer(max_models=20, max_runtime_secs=300)
    # trainer.prepare_data(processed, target_col='target')
    # trainer.train(target_col='target')

    # # Preparación para validación
    # X = processed.drop(columns='target')
    # y = processed['target']

    # # Split de datos
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, stratify=y, random_state=42
    # )

    # # Predicciones y evaluación
    # predictions = trainer.predict(X_test, auto_threshold=True, y_true=y_test)

    # current_file = Path(__file__).resolve()
    # project_root = current_file.parent.parent.parent
    # models_dir = project_root / 'models'
    # current_time = datetime.now().strftime("%Y-%m-%d_T_%H_%M_%S")
    # model_name = "h2o_automl_model" + f"_{current_time}"
    # h2o_model_path = str(models_dir) + f"/{model_name}/"
    # h2o_model_path = str(models_dir) + f"/h2o_automl_model/"

    # logger.info(f"Saving model to {models_dir} with name {model_name}")
    # trainer.save_model(model_name=model_name, base_path=str(models_dir))

    return preprocessor, sociodemographic_cols  # h2o_model_path
