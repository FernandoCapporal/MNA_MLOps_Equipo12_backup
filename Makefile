launch_api:
	uvicorn caravan_prediction_app.application.app:app --host 0.0.0.0 --port 8080 --workers ${API_WORKERS} --loop asyncio --log-level debug
