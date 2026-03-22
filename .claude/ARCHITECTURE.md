# SimpleInference — Architecture

> **Filosofía:** La simplicidad de Ultralytics/Inference + la flexibilidad de Triton, sin su complejidad.  
> **Caso de uso primario:** Inferencia en tiempo real — cámaras en vivo, drones, edge devices.

---

## Visión General

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleInference Server                       │
│                    (Docker Container)                           │
│                                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │   gRPC Server       │    │   FastAPI (HTTP)             │   │
│  │   port: 50051       │    │   port: 8080                 │   │
│  │                     │    │                              │   │
│  │  - Predict()        │    │  GET /config                 │   │
│  │  - StreamPredict()  │    │  GET /metrics                │   │
│  │  - GetConfig()      │    │  GET /health                 │   │
│  └────────┬────────────┘    └──────────────────────────────┘   │
│           │                                                     │
│  ┌────────▼────────────────────────────────────────────────┐   │
│  │                  ModelManager                           │   │
│  │                                                         │   │
│  │  - load_model()     - hot_swap()                        │   │
│  │  - warmup()         - get_active_model()                │   │
│  │  - get_config()                                         │   │
│  └────────┬────────────────────────────────────────────────┘   │
│           │                                                     │
│  ┌────────▼────────────────────────────────────────────────┐   │
│  │              BaseDetector (Abstract)                    │   │
│  │                                                         │   │
│  │  + preprocess(image) → tensor                           │   │
│  │  + predict(tensor)   → raw_output                       │   │
│  │  + postprocess(raw)  → sv.Detections                    │   │
│  │  + warmup(synthetic) → None                             │   │
│  └──────┬───────────────┬────────────────────────────────  ┘   │
│         │               │                                       │
│  ┌──────▼──────┐  ┌─────▼──────┐  ┌─────────────┐            │
│  │ YOLO11      │  │ RF-DETR    │  │  + Future   │            │
│  │ Detector    │  │ Detector   │  │  Detectors  │            │
│  └──────┬──────┘  └─────┬──────┘  └─────────────┘            │
│         │               │                                       │
│  ┌──────▼───────────────▼──────────────────────────────────┐   │
│  │                 Backend Layer                           │   │
│  │                                                         │   │
│  │   PyTorchBackend │ ONNXBackend │ TensorRTBackend         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Device Manager                             │   │
│  │   CPU (x86_64) │ CUDA │ [MPS - Fase 2]                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           ▲
           │  gRPC (protobuf)
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                    InferenceClient                              │
│                    (archivo .py portable)                       │
│                                                                 │
│  - __init__(host, port, config=None)                            │
│  - connect() → espera activa hasta que servidor esté listo      │
│  - fetch_config() → sincroniza con /config del servidor         │
│  - predict(image) → sv.Detections                              │
│  - stream_predict(frame_generator) → Iterator[sv.Detections]   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes del Servidor

### 1. gRPC Service (`detections.proto`)
- `Predict(InferenceRequest) → InferenceResponse` — unario, batch_size=1 por defecto
- `StreamPredict(stream InferenceRequest) → stream InferenceResponse` — bidireccional para video en tiempo real
- `GetServerConfig(Empty) → ServerConfigResponse` — expone la configuración activa

### 2. FastAPI HTTP Server
Corre en paralelo al gRPC en el mismo contenedor.

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado del servidor (listo/cargando) |
| `/config` | GET | Configuración activa: modelo, backend, dimensiones |
| `/metrics` | GET | Latencia, throughput, frames procesados, errores |

### 3. ModelManager
- Carga el modelo especificado en el YAML de configuración al arranque
- Soporta **hot-swap**: reemplazar el modelo activo sin reiniciar el servidor
- Gestiona el warmup con datos sintéticos antes de servir tráfico real
- Un solo modelo activo a la vez (Fase 1 y 2); concurrent execution en Fase 3

### 4. BaseDetector (Abstracción Central)
Clase abstracta que todos los detectores deben implementar:

```python
class BaseDetector(ABC):
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any: ...

    @abstractmethod
    def predict(self, tensor: Any) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any) -> sv.Detections: ...

    def warmup(self, n_iterations: int = 5) -> None: ...
```

**Principio clave:** el output de `postprocess()` es **siempre** `sv.Detections`, independientemente del modelo o backend.

### 5. Backend Layer
Abstracción sobre el runtime de inferencia:

| Backend | Dispositivos soportados | Cuándo usarlo |
|---|---|---|
| `PyTorchBackend` | CPU, CUDA | Desarrollo, flexibilidad |
| `ONNXBackend` | CPU, CUDA | Producción portable |
| `TensorRTBackend` | CUDA únicamente | Producción máxima performance |

### 6. Device Manager
Detección automática del dispositivo disponible con fallback:
`TensorRT → CUDA → ONNX+CUDA → ONNX+CPU → PyTorch+CPU`

---

## Componente Cliente

### InferenceClient (archivo único portable)
Diseñado para ser copiado a cualquier proyecto con cero dependencias adicionales más allá de `grpcio`, `opencv-python` y `supervision`.

**Flujo de uso:**
```python
client = InferenceClient(host="localhost", port=50051)
client.connect()          # espera activa hasta que el servidor responda
client.fetch_config()     # sincroniza width/height/modelo del servidor

frame = cv2.imread("frame.jpg")
detections = client.predict(frame)   # sv.Detections
```

**Flujo interno de `predict()`:**
1. Resize a las dimensiones del modelo (obtenidas del servidor vía `fetch_config()`)
2. Encode a JPEG con calidad configurable
3. Serializar en `InferenceRequest` protobuf
4. Enviar vía gRPC
5. Deserializar `InferenceResponse` → `sv.Detections`

---

## Configuración YAML

```yaml
# config.yaml — ejemplo completo
server:
  host: "0.0.0.0"
  grpc_port: 50051
  http_port: 8080
  max_workers: 4

model:
  name: "yolo11n"           # nombre del modelo
  version: "1.0.0"
  type: "yolo11"            # yolo11 | rfdetr
  backend: "onnx"           # pytorch | onnx | tensorrt
  source: "local"           # local | roboflow
  path: "/models/yolo11n.onnx"   # si source=local
  # roboflow_project: "my-project"  # si source=roboflow
  # roboflow_version: 3
  input_width: 640
  input_height: 640
  confidence_threshold: 0.5
  iou_threshold: 0.45
  class_names:
    - "person"
    - "car"

inference:
  batch_size: 1
  max_batch_size: 8         # para dynamic batching
  dynamic_batching: false

warmup:
  enabled: true
  iterations: 5
  synthetic_data: true      # genera imágenes aleatorias para warmup
```

---

## Carga de Modelos

El servidor sigue esta lógica de resolución al arrancar:

```
1. Leer config.yaml
2. Si source=local → buscar en path especificado
3. Si source=roboflow → descargar via Roboflow API
4. Si source=local y NO existe el archivo → intentar descarga automática
5. Warmup con datos sintéticos
6. Servidor listo para recibir tráfico
```

---

## Estructura de Directorios del Proyecto

```
simpleinference/
├── server/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── config.yaml                  # configuración por defecto
│   ├── requirements.txt
│   ├── main.py                      # entrypoint: lanza gRPC + FastAPI
│   ├── proto/
│   │   └── detections.proto
│   ├── generated/                   # archivos generados por protoc
│   │   ├── detections_pb2.py
│   │   └── detections_pb2_grpc.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   ├── device_manager.py
│   │   └── config_loader.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base_detector.py
│   │   ├── yolo11_detector.py
│   │   └── rfdetr_detector.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base_backend.py
│   │   ├── pytorch_backend.py
│   │   ├── onnx_backend.py
│   │   └── tensorrt_backend.py
│   ├── services/
│   │   ├── grpc_service.py          # implementación del servicer gRPC
│   │   └── http_service.py          # FastAPI: /health /config /metrics
│   └── tests/
│       ├── test_detectors.py
│       ├── test_backends.py
│       ├── test_grpc_service.py
│       └── test_config_loader.py
│
├── client/
│   ├── inference_client.py          # archivo único portable
│   └── tests/
│       ├── test_client.py
│       └── test_integration.py
│
├── models/                          # volumen Docker (gitignored)
│   └── .gitkeep
│
├── ARCHITECTURE.md
├── PLAN.md
├── PHASE1.md
├── PHASE2.md
├── PHASE3.md
├── CLAUDE.md
├── REFERENCES.md
└── JOURNAL.md
```

---

## Flujo de Datos: Tiempo Real

```
Cámara/Drone
    │
    ▼
InferenceClient.stream_predict(frame_generator)
    │  JPEG encode + protobuf
    ▼
gRPC StreamPredict (bidireccional)
    │
    ▼
InferenceServer.StreamPredict()
    │
    ▼
ModelManager.get_active_model()
    │
    ▼
BaseDetector.preprocess() → predict() → postprocess()
    │  sv.Detections
    ▼
Serializar → InferenceResponse protobuf
    │
    ▼
InferenceClient recibe → deserializa → sv.Detections
    │
    ▼
Tu aplicación (annotate, track, alert, etc.)
```

---

## Salida Estandarizada

Todo detector retorna `sv.Detections` de Roboflow Supervision:

```python
import supervision as sv

detections = sv.Detections(
    xyxy=np.array([[x1, y1, x2, y2], ...]),   # float32
    confidence=np.array([0.95, ...]),           # float32
    class_id=np.array([0, 1, ...]),            # int32
    data={"class_name": np.array(["person"])}  # opcional
)
```

Esto habilita directamente el uso de todas las utilidades de Supervision:
`sv.BoxAnnotator`, `sv.ByteTracker`, `detections.filter_by_confidence()`, etc.
