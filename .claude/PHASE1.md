# Fase 1 — Fundamentos

> **Meta:** Sistema funcional end-to-end con YOLO11n como primer detector.  
> **Principio de la fase:** Diseñar la abstracción correcta antes que la implementación completa.  
> **Duración estimada:** 2–3 semanas

---

## Objetivos

1. Definir e implementar `BaseDetector` — el contrato que todos los detectores deben cumplir
2. Implementar `YOLO11Detector` como primera implementación concreta
3. Configurar el servidor gRPC funcional con `Predict()` unario
4. Configurar FastAPI con `/health` y `/config`
5. Implementar `InferenceClient` portable con espera activa
6. Sistema de configuración YAML
7. Docker Compose operativo
8. Suite completa de unit tests

---

## Entregables

| Entregable | Descripción |
|---|---|
| `detections.proto` actualizado | Incluye `GetServerConfig` RPC |
| `base_detector.py` | Clase abstracta con contrato completo |
| `yolo11_detector.py` | Implementación con PyTorch y ONNX |
| `pytorch_backend.py` | Backend PyTorch |
| `onnx_backend.py` | Backend ONNX |
| `model_manager.py` | Carga, warmup, exposición del modelo activo |
| `device_manager.py` | Detección automática CPU/CUDA |
| `config_loader.py` | Parser de config.yaml con validación |
| `grpc_service.py` | Servicer gRPC: `Predict` + `GetServerConfig` |
| `http_service.py` | FastAPI: `/health` + `/config` |
| `main.py` | Entrypoint que lanza gRPC + FastAPI concurrentemente |
| `inference_client.py` | Cliente portable completo |
| `Dockerfile` + `docker-compose.yml` | Infraestructura de contenedor |
| `config.yaml` | Configuración por defecto |
| Tests unitarios | Cobertura >= 80% del código de Fase 1 |

---

## Tareas Detalladas

### T1.1 — Proto actualizado
Actualizar `detections.proto` para incluir:
- `StreamPredict` (definido pero implementado en Fase 2)
- `GetServerConfig` RPC
- `ServerConfigResponse` message con campos: model_name, model_type, backend, input_width, input_height, version

### T1.2 — Config Loader
```yaml
# Validaciones requeridas:
# - backend debe ser: pytorch | onnx | tensorrt
# - model.type debe ser: yolo11 | rfdetr
# - source debe ser: local | roboflow
# - Si source=local, path es requerido
# - Si source=roboflow, roboflow_project y roboflow_version son requeridos
```
- Leer `config.yaml`
- Validar campos obligatorios
- Aplicar valores por defecto donde corresponde
- Exponer configuración como dataclass tipada

### T1.3 — Device Manager
- Detectar si CUDA está disponible (`torch.cuda.is_available()`)
- Detectar si `onnxruntime-gpu` está instalado
- Devolver el dispositivo óptimo según lo que esté disponible
- Fallback automático: TensorRT → CUDA → ONNX+CUDA → ONNX+CPU → PyTorch+CPU

### T1.4 — Backend Layer (PyTorch + ONNX)
```python
class BaseBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, device: str) -> None: ...

    @abstractmethod
    def infer(self, tensor: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def is_available(cls) -> bool: ...
```

### T1.5 — BaseDetector + YOLO11Detector
```python
class BaseDetector(ABC):
    def __init__(self, config: ModelConfig, backend: BaseBackend): ...

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalize, transpose según el modelo."""
        ...

    @abstractmethod
    def postprocess(self, raw_output: np.ndarray, original_shape: tuple) -> sv.Detections:
        """NMS (si aplica), filtrado por confianza, escalar bboxes."""
        ...

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Pipeline completo: preprocess → backend.infer → postprocess."""
        tensor = self.preprocess(image)
        raw = self.backend.infer(tensor)
        return self.postprocess(raw, image.shape[:2])

    def warmup(self, n_iterations: int = 5) -> None:
        """Genera imágenes sintéticas y corre predict() n veces."""
        synthetic = np.random.randint(0, 255,
            (self.config.input_height, self.config.input_width, 3),
            dtype=np.uint8)
        for _ in range(n_iterations):
            self.predict(synthetic)
```

**YOLO11Detector** — consideraciones de implementación:
- Usar `ultralytics` para cargar el modelo
- El preprocessing incluye: resize, BGR→RGB, normalización [0,1], HWC→CHW
- El postprocessing incluye: filtrado por `confidence_threshold`, escalar coordenadas al tamaño original

### T1.6 — Model Manager
```python
class ModelManager:
    def __init__(self, config: ServerConfig): ...
    def load_model(self) -> None: ...        # carga según config
    def warmup(self) -> None: ...            # delega a detector.warmup()
    def get_active_model(self) -> BaseDetector: ...
    def hot_swap(self, new_config: ModelConfig) -> None: ...  # Fase 2
    def get_model_info(self) -> dict: ...    # para /config endpoint
```

**Lógica de carga:**
1. Si `source=local` → verificar que el archivo existe en `path`
2. Si no existe (o `source=roboflow`) → descargar con `roboflow` SDK
3. Instanciar backend correcto según config
4. Instanciar detector correcto según `model.type`
5. Llamar `warmup()` antes de marcar el servidor como listo

### T1.7 — gRPC Service
```python
class InferenceServicer(detections_pb2_grpc.InferenceServiceServicer):
    def Predict(self, request, context) -> InferenceResponse:
        # 1. Deserializar imagen del request
        # 2. model_manager.get_active_model().predict(img)
        # 3. Serializar sv.Detections → InferenceResponse
        ...

    def GetServerConfig(self, request, context) -> ServerConfigResponse:
        # Delegar a model_manager.get_model_info()
        ...
```

### T1.8 — HTTP Service (FastAPI)
```python
@app.get("/health")
def health() -> dict:
    # {"status": "ready" | "loading", "model": "yolo11n", "uptime_seconds": 42}

@app.get("/config")
def config() -> dict:
    # Retorna la configuración completa del modelo activo
```

### T1.9 — Main Entrypoint
- Lanzar gRPC server en un thread
- Lanzar FastAPI con uvicorn en el thread principal
- Manejar señales de shutdown gracefully (SIGTERM para Docker)

### T1.10 — InferenceClient
```python
class InferenceClient:
    def __init__(self,
                 host: str = "localhost",
                 port: int = 50051,
                 http_port: int = 8080,
                 jpeg_quality: int = 85,
                 timeout_seconds: float = 30.0):
        ...

    def connect(self) -> None:
        """
        Espera activa: hace polling a /health cada 2s hasta que
        el servidor responda {"status": "ready"} o se agote timeout.
        Lanza ConnectionError si timeout.
        """
        ...

    def fetch_config(self) -> dict:
        """
        Llama a /config del servidor HTTP.
        Actualiza internamente self._input_width y self._input_height.
        """
        ...

    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        1. Resize a (self._input_width, self._input_height)
        2. Encode JPEG con self._jpeg_quality
        3. Enviar InferenceRequest via gRPC
        4. Deserializar respuesta → sv.Detections
        """
        ...
```

### T1.11 — Docker
**Dockerfile** (servidor):
- Base: `python:3.11-slim` (CPU) o `nvidia/cuda:12.x-runtime` (GPU)
- Compilar proto en build stage
- Copiar modelos desde volumen en runtime

**docker-compose.yml**:
```yaml
services:
  inference-server:
    build: .
    ports:
      - "50051:50051"
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./config.yaml:/app/config.yaml
    environment:
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY}
    # GPU opcional:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

---

## Tests de Fase 1

### `test_config_loader.py`
- [ ] Config válido carga correctamente
- [ ] Config con campos faltantes lanza `ValueError`
- [ ] Valores por defecto se aplican correctamente
- [ ] Backend inválido lanza error descriptivo

### `test_device_manager.py`
- [ ] Detecta CPU siempre como fallback
- [ ] Fallback funciona cuando CUDA no está disponible

### `test_backends.py`
- [ ] `ONNXBackend.is_available()` no lanza excepciones
- [ ] `PyTorchBackend` carga un modelo tiny y produce output con shape correcto

### `test_detectors.py`
- [ ] `YOLO11Detector.preprocess()` produce tensor de shape correcto `(1, 3, H, W)`
- [ ] `YOLO11Detector.postprocess()` retorna `sv.Detections` válido
- [ ] `warmup()` completa sin errores con datos sintéticos
- [ ] `predict()` con imagen real retorna `sv.Detections` (usando YOLO11n)

### `test_grpc_service.py`
- [ ] `Predict()` con imagen sintética retorna `InferenceResponse` con estructura correcta
- [ ] `GetServerConfig()` retorna la config activa
- [ ] Manejo correcto de imagen corrupta (context.abort con código apropiado)

### `test_client.py`
- [ ] `connect()` levanta `ConnectionError` si el servidor no responde en timeout
- [ ] `fetch_config()` actualiza dimensiones internas correctamente
- [ ] `predict()` retorna `sv.Detections` cuando el servidor está disponible
- [ ] Imagen de distinto tamaño al modelo se resizea correctamente antes de enviar

---

## Definición de "Fase 1 Completa"

- [ ] `docker-compose up` levanta el servidor sin errores
- [ ] `client.connect()` conecta exitosamente
- [ ] `client.fetch_config()` sincroniza dimensiones del modelo
- [ ] `client.predict(image)` retorna `sv.Detections` con detecciones reales de YOLO11n
- [ ] `/health` retorna `{"status": "ready"}`
- [ ] `/config` retorna la configuración del modelo activo
- [ ] Warmup completa antes de que el servidor sirva tráfico
- [ ] Todos los unit tests pasan
- [ ] JOURNAL.md actualizado con los cambios relevantes de la fase
