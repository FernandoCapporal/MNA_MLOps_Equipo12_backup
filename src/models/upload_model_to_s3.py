import os
import boto3
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def upload_h2o_model_to_s3(local_model_path, bucket_name, s3_folder="h2o_models", aws_region="us-east-1"):
    """
    Sube un modelo H2O completo a S3.

    Args:
        local_model_path: Ruta absoluta local al directorio del modelo H2O
        bucket_name: Nombre del bucket S3
        s3_folder: Carpeta destino en S3 (por defecto: h2o_models)
        aws_region: Región de AWS
    """

    # Verificar que las credenciales de AWS están disponibles
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not all([aws_access_key, aws_secret_key]):
        logger.error("Credenciales de AWS no encontradas en variables de entorno")
        logger.error("Por favor, configura AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY")
        return False

    # Verificar que la ruta local existe
    if not os.path.exists(local_model_path):
        logger.error(f"La ruta local no existe: {local_model_path}")
        return False

    if not os.path.isdir(local_model_path):
        logger.error(f"La ruta debe ser un directorio: {local_model_path}")
        return False

    try:
        # Configurar cliente S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Verificar que el bucket existe
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket S3 verificado: {bucket_name}")

        # Obtener nombre del modelo desde la ruta
        model_name = os.path.basename(local_model_path.rstrip('/'))
        s3_base_path = f"{s3_folder}/{model_name}"

        logger.info(f"Iniciando subida del modelo: {model_name}")
        logger.info(f"Ruta local: {local_model_path}")
        logger.info(f"Destino S3: s3://{bucket_name}/{s3_base_path}/")

        # Contadores para estadísticas
        total_files = 0
        uploaded_files = 0

        # Subir todos los archivos del directorio del modelo
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_file_path = os.path.join(root, file)

                # Calcular ruta relativa para mantener estructura
                relative_path = os.path.relpath(local_file_path, local_model_path)
                s3_key = f"{s3_base_path}/{relative_path}"

                try:
                    # Subir archivo
                    s3_client.upload_file(local_file_path, bucket_name, s3_key)
                    uploaded_files += 1
                    logger.debug(f"Subido: {relative_path}")

                except Exception as e:
                    logger.error(f"Error subiendo {relative_path}: {e}")
                    return False

                total_files += 1

        logger.info(f" Subida completada: {uploaded_files}/{total_files} archivos")
        logger.info(f" Modelo disponible en: s3://{bucket_name}/{s3_base_path}/")

        # Devolver la ruta S3 del modelo para referencia futura
        return f"s3://{bucket_name}/{s3_base_path}"

    except Exception as e:
        logger.error(f"Error en la subida a S3: {e}")
        return False


def get_environment_variables():
    """
    Obtiene y valida las variables de entorno necesarias.

    Returns:
        tuple: (local_model_path, bucket_name, s3_folder, aws_region) o (None, None, None, None) si hay error
    """
    # Variables requeridas
    local_model_path = os.getenv("H2O_MODEL_PATH")
    bucket_name = os.getenv("S3_BUCKET_NAME")

    if not local_model_path:
        logger.error("Variable de entorno H2O_MODEL_PATH no configurada")
        return None, None, None, None

    if not bucket_name:
        logger.error("Variable de entorno S3_BUCKET_NAME no configurada")
        return None, None, None, None

    # Variables opcionales con valores por defecto
    s3_folder = os.getenv("S3_MODELS_FOLDER", "h2o_models")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Convertir a ruta absoluta
    absolute_model_path = os.path.abspath(local_model_path)

    return absolute_model_path, bucket_name, s3_folder, aws_region


def main():
    """Función principal que usa variables de entorno."""

    logger.info("🚀 Iniciando subida de modelo H2O a S3 usando variables de entorno")

    # Obtener variables de entorno
    local_model_path, bucket_name, s3_folder, aws_region = get_environment_variables()

    if not all([local_model_path, bucket_name]):
        logger.error(" No se pudieron obtener todas las variables de entorno requeridas")
        logger.info(" Variables de entorno requeridas:")
        logger.info("   H2O_MODEL_PATH: Ruta al directorio del modelo H2O")
        logger.info("   S3_BUCKET_NAME: Nombre del bucket S3")
        logger.info(" Variables de entorno opcionales:")
        logger.info("   S3_MODELS_FOLDER: Carpeta en S3 (default: h2o_models)")
        logger.info("   AWS_DEFAULT_REGION: Región de AWS (default: us-east-1)")
        logger.info("   AWS_ACCESS_KEY_ID: Credencial de AWS")
        logger.info("   AWS_SECRET_ACCESS_KEY: Credencial de AWS")
        exit(1)

    # Verificar credenciales AWS
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not all([aws_access_key, aws_secret_key]):
        logger.error("    Credenciales AWS no encontradas")
        logger.error("   Configura AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY")
        exit(1)

    # Mostrar configuración
    logger.info("📋 Configuración:")
    logger.info(f"   Modelo local: {local_model_path}")
    logger.info(f"   Bucket S3: {bucket_name}")
    logger.info(f"   Carpeta S3: {s3_folder}")
    logger.info(f"   Región AWS: {aws_region}")

    # Ejecutar subida
    success = upload_h2o_model_to_s3(
        local_model_path=local_model_path,
        bucket_name=bucket_name,
        s3_folder=s3_folder,
        aws_region=aws_region
    )

    if success:
        logger.info("Modelo H2O subido exitosamente a S3")
    else:
        logger.error("Error subiendo modelo a S3")
        exit(1)


if __name__ == "__main__":
    main()