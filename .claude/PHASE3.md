# Fase 3 — Producción

> **Meta:** Sistema production-ready con observabilidad completa, dynamic batching y concurrent model execution.  
> **Prerrequisito:** Fase 2 completamente funcional y tests al 100%.  
> **Duración estimada:** 2–3 semanas

---

## Objetivos

1. `/metrics` endpoint con latencia P50/P95/P99, throughput y errores
2. Dynamic batching configurable
3. Concurrent model execution (2+ modelos simultáneos)
4. Tests de carga y benchmarks
5. Documentación de usuario final

---

## Entregables

| Entregable | Descripción |
|---|---|
| `metrics_collector.py` | Recolector de métricas en memoria |
| `http_service.py` actualizado | `/metrics` endpoint completo |
| `batch_manager.py` | Dynamic batching con timeout de ventana |
| `model_manager.py` actualizado | Concurrent execution con pool de modelos |
| `docker-compose.yml` actualizado | Configuración de producción con health checks |
| Tests de carga | `test_load.py` con 30fps sostenido |
| `README.md` | Documentación de usuario final |

---

## Tareas Detalladas

### T3.1 — Sistema de Métricas

```python
class MetricsCollector:
    """
    Recolector de métricas en memoria (sin dependencias externas).
    Usa sliding window de últimos N segundos para calcular percentiles.
    """
    def record_inference(self, latency_ms: float, batch_size: int) -> None: ...
    def record_error(self, error_type: str) -> None: ...
    def get_metrics(self) -> dict: ...
```

**`GET /metrics` — respuesta esperada:**
```json
{
  "model": "yolo11n",
  "backend": "onnx",
  "device": "cuda:0",
  "uptime_seconds": 3600,
  "inference": {
    "total_requests": 108000,
    "total_errors": 3,
    "error_rate_percent": 0.003,
    "throughput_fps": 30.2,
    "latency_ms": {
      "p50": 12.4,
      "p95": 18.7,
      "p99": 24.1,
      "min": 8.2,
      "max": 45.6
    }
  },
  "batching": {
    "enabled": false,
    "current_batch_size": 1,
    "avg_batch_size": 1.0
  },
  "memory": {
    "gpu_used_mb": 512,
    "gpu_total_mb": 8192
  }
}
```

### T3.2 — Dynamic Batching

Agrupa requests individuales en batches para maximizar throughput en GPU.

```python
class BatchManager:
    def __init__(self,
                 max_batch_size: int = 8,
                 window_timeout_ms: float = 10.0):
        """
        Acumula requests durante window_timeout_ms o hasta max_batch_size,
        lo que ocurra primero.
        """
        ...

    async def submit(self, image: np.ndarray) -> sv.Detections:
        """
        Submete una imagen y espera el resultado del batch.
        El BatchManager decide cuándo procesar el batch acumulado.
        """
        ...
```

**Habilitación por config:**
```yaml
inference:
  dynamic_batching: true
  max_batch_size: 8
  batch_window_ms: 10.0   # máximo tiempo de espera antes de procesar batch incompleto
```

**Nota:** Dynamic batching solo tiene sentido en GPU. En CPU con batch_size=1 es más rápido.

### T3.3 — Concurrent Model Execution

Permite tener múltiples modelos cargados simultáneamente y enrutar requests al modelo correcto.

```yaml
# config.yaml — modo concurrent
models:
  - name: "detector"
    type: "yolo11"
    path: "/models/yolo11n.onnx"
    backend: "onnx"
    input_width: 640
    input_height: 640

  - name: "classifier"
    type: "yolo11"         # YOLO11 en modo clasificación
    path: "/models/yolo11n-cls.onnx"
    backend: "onnx"
    input_width: 224
    input_height: 224
```

**Proto actualizado para Fase 3:**
```protobuf
message InferenceRequest {
  bytes image_data = 1;
  int32 width = 2;
  int32 height = 3;
  string model_name = 4;   // NUEVO: nombre del modelo destino (vacío = modelo por defecto)
}
```

**ModelManager en modo concurrent:**
```python
class ModelManager:
    def __init__(self, config):
        self._models: Dict[str, BaseDetector] = {}   # pool de modelos

    def get_model(self, name: str = "") -> BaseDetector:
        """Retorna el modelo por nombre o el primero si name está vacío."""
        ...
```

### T3.4 — Tests de Carga

```python
# test_load.py
def test_sustained_30fps_streaming():
    """
    Envía frames a 30fps durante 60 segundos.
    Criterio de éxito:
    - Latencia P99 < 50ms
    - 0 errores
    - Throughput >= 28fps promedio (margen de 7%)
    """
    ...

def test_concurrent_clients():
    """
    5 clientes simultáneos enviando requests.
    Criterio: todos reciben respuestas válidas, sin deadlocks.
    """
    ...

def test_memory_stability():
    """
    10,000 inferencias consecutivas.
    Criterio: uso de memoria GPU no crece más del 5%.
    """
    ...
```

### T3.5 — Hardening del Servidor

- **Graceful shutdown:** al recibir SIGTERM, terminar requests en curso antes de apagar
- **Request timeout:** requests que superen X segundos retornan `DEADLINE_EXCEEDED`
- **Input validation:** imágenes corruptas o de dimensiones inválidas retornan `INVALID_ARGUMENT` con mensaje descriptivo
- **Memory limits:** si la GPU llega al 95% de uso, retornar `RESOURCE_EXHAUSTED`

### T3.6 — Docker Producción

```yaml
# docker-compose.yml — configuración de producción
services:
  inference-server:
    build:
      context: .
      target: production
    restart: unless-stopped
    ports:
      - "50051:50051"
      - "8080:8080"
    volumes:
      - ./models:/models:ro          # read-only en producción
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s             # tiempo para warmup
    deploy:
      resources:
        limits:
          memory: 4G
        # GPU (opcional):
        # reservations:
        #   devices:
        #     - driver: nvidia
        #       count: 1
        #       capabilities: [gpu]
```

---

## Tests de Fase 3

### `test_metrics.py`
- [ ] `MetricsCollector` registra latencias correctamente
- [ ] Percentiles P50/P95/P99 son estadísticamente correctos (vs valores conocidos)
- [ ] `/metrics` retorna JSON con estructura completa
- [ ] Métricas se resetean correctamente después de hot-swap
- [ ] Error rate se calcula correctamente

### `test_batching.py`
- [ ] `BatchManager` agrupa hasta `max_batch_size` requests
- [ ] `window_timeout_ms` dispara procesamiento de batch incompleto
- [ ] Batch_size=1 (default) funciona igual que sin batching
- [ ] Throughput con batching >= throughput sin batching en GPU (benchmark)

### `test_concurrent_models.py`
- [ ] Dos modelos cargados simultáneamente retornan resultados independientes
- [ ] Request con `model_name` específico enruta al modelo correcto
- [ ] Request sin `model_name` usa el modelo por defecto
- [ ] Hot-swap de un modelo no afecta al otro

### `test_load.py`
- [ ] 30fps sostenido por 60s: latencia P99 < 50ms
- [ ] 5 clientes concurrentes: 0 errores en 1000 requests totales
- [ ] 10,000 inferencias: memoria GPU estable (delta < 5%)

---

## Definición de "Fase 3 Completa"

- [ ] `/metrics` expone P50/P95/P99 en tiempo real
- [ ] Dynamic batching habilitado y validado en benchmark
- [ ] Concurrent model execution funcional con 2 modelos
- [ ] Tests de carga pasan: 30fps × 60s sin errores
- [ ] Dockerfile con multi-stage build (imagen reducida)
- [ ] `README.md` con guía de inicio rápido, ejemplos de uso y referencia de configuración
- [ ] JOURNAL.md actualizado con todos los cambios relevantes de la fase
