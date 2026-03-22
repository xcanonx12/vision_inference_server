"""FastAPI HTTP service for health, config, and metrics endpoints."""

import logging
import time

from fastapi import FastAPI

from server.core.model_manager import ModelManager

logger = logging.getLogger(__name__)


def create_app(model_manager: ModelManager) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_manager: The active ModelManager instance.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(title="SimpleInference", version="1.0.0")
    start_time = time.time()

    @app.get("/health")
    def health() -> dict:
        status = "ready" if model_manager.is_ready() else "loading"
        info = model_manager.get_model_info()
        return {
            "status": status,
            "model": info.get("model_name", ""),
            "uptime_seconds": round(time.time() - start_time, 1),
        }

    @app.get("/config")
    def config() -> dict:
        return model_manager.get_model_info()

    return app
