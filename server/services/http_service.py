"""FastAPI HTTP service for health, config, and metrics endpoints."""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.core.config_loader import ModelConfig
from server.core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class HotSwapRequest(BaseModel):
    name: str
    type: str
    backend: str = "pytorch"
    source: str = "local"
    path: Optional[str] = None
    version: str = "1.0.0"
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    trt_fp16: bool = False


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

    @app.post("/hot-swap")
    def hot_swap(req: HotSwapRequest) -> dict:
        new_cfg = ModelConfig(
            name=req.name,
            type=req.type,
            backend=req.backend,
            source=req.source,
            path=req.path,
            version=req.version,
            input_width=req.input_width,
            input_height=req.input_height,
            confidence_threshold=req.confidence_threshold,
            iou_threshold=req.iou_threshold,
            trt_fp16=req.trt_fp16,
        )
        try:
            model_manager.hot_swap(new_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"status": "swapped", "model": req.name}

    return app
