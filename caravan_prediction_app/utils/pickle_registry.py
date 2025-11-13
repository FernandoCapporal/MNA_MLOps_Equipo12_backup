# caravan_prediction_app/utils/pickle_registry.py
import sys
import __main__
from caravan_prediction_app.application.settings import Settings
import logging

settings = Settings()
logger = logging.getLogger(settings.APPLICATION_ID)


def register_custom_classes():
    """Registra las clases custom para pickle de forma segura"""
    try:
        from src.pipelines.build_pipeline import (
            SociodemographicToZoneTransformer,
            ColumnDropper,
            SkewnessCorrector,
            H2OPredictor,
        )

        __main__.SociodemographicToZoneTransformer = SociodemographicToZoneTransformer
        __main__.ColumnDropper = ColumnDropper
        __main__.SkewnessCorrector = SkewnessCorrector
        __main__.H2OPredictor = H2OPredictor

        import inspect
        current_frame = inspect.currentframe()
        if current_frame:
            caller_globals = current_frame.f_back.f_globals
            caller_globals['SociodemographicToZoneTransformer'] = SociodemographicToZoneTransformer
            caller_globals['ColumnDropper'] = ColumnDropper
            caller_globals['SkewnessCorrector'] = SkewnessCorrector
            caller_globals['H2OPredictor'] = H2OPredictor

        editable_modules = ['__main__', '__mp_main__', 'builtins']
        for module_name in editable_modules:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                try:
                    setattr(module, 'SociodemographicToZoneTransformer', SociodemographicToZoneTransformer)
                    setattr(module, 'ColumnDropper', ColumnDropper)
                    setattr(module, 'SkewnessCorrector', SkewnessCorrector)
                    setattr(module, 'H2OPredictor', H2OPredictor)
                except (TypeError, AttributeError):
                    # Algunos módulos no permiten modificación, los saltamos
                    pass

        logger.info("Clases custom registradas exitosamente para pickle")
        return True

    except ImportError as e:
        logger.error(f"Error registrando clases: {e}")
        return False
