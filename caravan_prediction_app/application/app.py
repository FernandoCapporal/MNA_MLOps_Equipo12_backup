import sys
import pytz
import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from splunk_otel.tracing import start_tracing
from datetime import datetime
from caravan_prediction_app.application.settings import Settings
from caravan_prediction_app.clases.inference_clases import PredictionInput
from caravan_prediction_app.services.insurance_company_server import PipelineSingleton, load_and_format_dataframe, \
    load_and_format_json_dataframe
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import Response
from io import BytesIO
from fastapi.responses import JSONResponse
from caravan_prediction_app.utils.pickle_registry import register_custom_classes
from src.pipelines.build_pipeline import (
    SociodemographicToZoneTransformer,
    ColumnDropper,
    SkewnessCorrector,
    H2OPredictor,
)

settings = Settings()
logger = logging.getLogger(settings.APPLICATION_ID)
pipeline_singleton = PipelineSingleton()

pipeline_path = 'models/'

start_tracing(service_name=settings.X_APPLICATION_ID)
logging.basicConfig(format=settings.LOG_PATTERN, level=settings.LOG_LEVEL)

app = FastAPI(
    title="Caravan Prediction API",
    description="API for inference in new data",
    version=settings.VERSION,
    openapi_tags=[
        {
            "name": "Health Check",
            "description": "Health check endpoint"
        },
        {
            "name": "Pipeline",
            "description": "Inference in caravan prediction"
        }
    ]
)

FastAPIInstrumentor.instrument_app(app)

local_tz = pytz.timezone(settings.TIMEZONE)
startup_time = datetime.now(tz=local_tz)
logger.info(f"Service {settings.APPLICATION_ID} started at {startup_time.isoformat()}")


# Logger configuration that aligns with Gunicorn's logging
def setup_logging():
    gunicorn_logger = logging.getLogger("gunicorn.error")

    if gunicorn_logger.handlers:
        logger.handlers = gunicorn_logger.handlers
        logger.setLevel(gunicorn_logger.level)
    else:
        # Logging for local development
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    # Register custom classes for pickle
    register_custom_classes()
    # Load the model from s3
    pipeline_singleton.load_model()
    # Load the pipeline in the singleton
    pipeline_singleton.load_pipeline(folder_name=pipeline_path)
    yield
    # === SHUTDOWN ===


app.router.lifespan_context = lifespan


@app.get(f"{settings.BASE_PATH}/health_check", tags=["Health Check"])
async def health_check():
    return {
        "app": f'{settings.APPLICATION_ID}:{settings.VERSION}',
        "status": "Up and running",
        "started_at": startup_time.isoformat(),
        "timezone": settings.TIMEZONE,
    }


@app.post(f"{settings.BASE_PATH}/predict", tags=["Pipeline"])
async def upload_csv(
        file: UploadFile = File(...)
):
    """
    Endpoint que recibe un archivo CSV, realiza inferencias
    y devuelve CSV con las predicciones
    """
    logger.info(f"Processing inference with uploaded CSV")

    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

        contents = await file.read()
        df = load_and_format_dataframe(contents)

        logger.info(f"CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        predictions_df = pipeline_singleton.predict(df)
        logger.info(f"Predictions generated successfully: {predictions_df.shape}")

        output = BytesIO()
        predictions_df.to_csv(output, index=False)
        output.seek(0)

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=inference_results.csv"
            }
        )

    except Exception as e:
        logger.error(f"Failed to process inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.BASE_PATH}/predict-json", tags=["Pipeline"])
async def predict_from_json(input_data: PredictionInput):
    """
    Endpoint que recibe una lista de diccionarios JSON,
    convierte a DataFrame, ejecuta inferencia y devuelve las predicciones como JSON.
    """
    logger.info("Processing inference with uploaded JSON")

    try:
        df = load_and_format_json_dataframe(input_data.data)

        predictions_df = pipeline_singleton.predict(df)
        logger.info(f"Predictions generated successfully: {predictions_df.shape}")

        return JSONResponse(
            content=predictions_df.to_dict(orient="records")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process inference from JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting server with Uvicorn")
    uvicorn.run(app, host=settings.LOCAL_HOST, port=settings.PORT, loop='asyncio')
    gunicorn_logger = logging.getLogger('gunicorn.error')
    logger.handlers = gunicorn_logger.handlers
    logger.setLevel(logging.DEBUG)
    logger.setLevel('DEBUG')
