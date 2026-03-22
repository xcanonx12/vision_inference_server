"""SimpleInference server entrypoint.

Launches gRPC and FastAPI servers concurrently with graceful shutdown.
"""

import logging
import signal
import sys
import threading
from concurrent import futures

import grpc
import uvicorn

from server.core.config_loader import load_config
from server.core.device_manager import DeviceManager
from server.core.metrics_collector import MetricsCollector
from server.core.model_manager import ModelManager
from server.generated import detections_pb2_grpc
from server.services.grpc_service import InferenceServicer
from server.services.http_service import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def serve(config_path: str = "server/config.yaml") -> None:
    """Start the inference server.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)

    # Device info
    device_manager = DeviceManager()
    logger.info("Device: %s", device_manager.get_optimal_device())

    # Load model
    model_manager = ModelManager(config)
    logger.info(
        "Loading model: %s (%s/%s)",
        config.model.name, config.model.type, config.model.backend,
    )
    model_manager.load_model()

    if config.warmup.enabled:
        logger.info("Warming up (%d iterations)...", config.warmup.iterations)
        model_manager.warmup()

    logger.info("Model ready.")

    # Metrics collector
    metrics_collector = MetricsCollector(window_seconds=60)

    # gRPC server
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.server.max_workers)
    )
    servicer = InferenceServicer(model_manager, metrics_collector)
    detections_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, grpc_server)
    grpc_server.add_insecure_port(
        f"{config.server.host}:{config.server.grpc_port}"
    )

    # FastAPI app
    app = create_app(model_manager, metrics_collector)

    # Graceful shutdown
    def handle_signal(signum: int, frame) -> None:
        logger.info("Shutdown signal received (signal %d).", signum)
        grpc_server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Start gRPC in background thread
    grpc_server.start()
    logger.info(
        "gRPC server listening on %s:%d",
        config.server.host, config.server.grpc_port,
    )

    # Start FastAPI in main thread (blocking)
    logger.info(
        "HTTP server listening on %s:%d",
        config.server.host, config.server.http_port,
    )
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.http_port,
        log_level="info",
    )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "server/config.yaml"
    serve(config_path)
