from pydantic import BaseSettings


class Settings(BaseSettings):
    # CONFIGS
    APPLICATION_ID: str = 'caravan-prediction'
    BASE_PATH: str = f"/api/{APPLICATION_ID}"
    VERSION: str = "1.0.0"
    PORT: int = 8080
    LOCAL_HOST = '0.0.0.0'
    COUNTRY: str = 'mx'
    ENVIRONMENT: str = 'LOCAL'
    TIMEZONE: str = 'America/Mexico_City'
    X_APPLICATION_ID: str = APPLICATION_ID
    LOG_PATTERN: str = '%(asctime)s.%(msecs)s:%(name)s:%(thread)d:(%(threadName)-10s):%(levelname)s:%(process)d:%(message)s'
    LOG_LEVEL: str = "INFO"
    BEST_THRESHOLD: float = None

    # S3
    AWS_ACCESS_KEY_ID: str = ''
    AWS_SECRET_ACCESS_KEY: str = ''
    S3_BUCKET_NAME: str = ''
    AWS_DEFAULT_REGION: str = ''

    class Config:
        case_sensitive = False
        env_file = '.env'
        env_file_encoding = 'utf-8'
