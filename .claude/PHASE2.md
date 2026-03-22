# Fase 2 — Completitud

> **Meta:** Sistema completo con todos los modelos, backends, streaming bidireccional y hot-swap.  
> **Prerrequisito:** Fase 1 completamente funcional y tests al 100%.  
> **Duración estimada:** 2–3 semanas

---

## Objetivos

1. Implementar `RF-DETRDetector` con el mismo contrato que `YOLO11Detector`
2. Implementar `TensorRTBackend` para CUDA
3. Streaming bidireccional gRPC (`StreamPredict`)
4. Hot-swap de modelos sin reiniciar el servidor
5. Soporte de `stream_predict()` en el cliente
6. Tests de integración completos

---

## Entregables

| Entregable | Descripción |
|---|---|
| `rfdetr_detector.py` | RF-DETR con PyTorch y ONNX backends |
| `tensorrt_backend.py` | Backend TensorRT (CUDA únicamente) |
| `grpc_service.py` actualizado | `StreamPredict` bidireccional implementado |
| `http_service.py` actualizado | Endpoint `POST /hot-swap` |
| `model_manager.py` actualizado | `hot_swap()` completo y thread-safe |
| `inference_client.py` actualizado | `stream_predict()` con frame generator |
| Tests de integración | Server + client end-to-end |

---

## Tareas Detalladas

### T2.1 — RF-DETRDetector

RF-DETR es un transformer-based detector sin NMS (Non-Maximum Suppression). Esto implica diferencias importantes respecto a YOLO11:

**Diferencias clave en preprocessing:**
- RF-DETR espera imágenes normalizadas con media/std ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- El tamaño de entrada puede variar (configurable, tipicamente 560 o 640)

**Diferencias clave en postprocessing:**
- No requiere NMS (el modelo ya produce detecciones únicas)
- Las coordenadas de salida son relativas `[0, 1]` → requiere escalar al tamaño original
- Filtrado solo por `confidence_threshold`

```python
class RFDETRDetector(BaseDetector):
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # BGR → RGB, resize, normalize con ImageNet stats, HWC → CHW
        ...

    def postprocess(self, raw_output: np.ndarray, original_shape: tuple) -> sv.Detections:
        # Filtrar por confianza, escalar coords relativas → absolutas
        # Retornar sv.Detections (sin NMS)
        ...
```

### T2.2 — TensorRT Backend

```python
class TensorRTBackend(BaseBackend):
    @classmethod
    def is_available(cls) -> bool:
        # Verificar que tensorrt y pycuda están instalados Y hay CUDA
        ...

    def load(self, model_path: str, device: str) -> None:
        # Cargar engine .trt desde archivo
        # Crear execution context
        # Allocar buffers en GPU
        ...

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        # Copiar input a GPU buffer
        # Ejecutar engine
        # Copiar output de GPU → CPU
        ...
```

**Notas importantes:**
- Los engines TensorRT son específicos del hardware donde se compilan
- El YAML debe permitir especificar si el archivo es `.onnx` (para compilar) o `.trt` (ya compilado)
- Añadir campo `trt_fp16: true/false` en config para precisión mixta

### T2.3 — Streaming Bidireccional gRPC

```protobuf
// Ya definido en proto, implementar en Fase 2:
rpc StreamPredict (stream InferenceRequest) returns (stream InferenceResponse) {}
```

```python
class InferenceServicer(...):
    def StreamPredict(self, request_iterator, context):
        for request in request_iterator:
            # Mismo pipeline que Predict(), pero en loop
            img = deserialize_image(request)
            detections = self.model_manager.get_active_model().predict(img)
            yield serialize_detections(detections)
```

**Consideraciones para tiempo real:**
- No acumular frames: procesar cada frame tan pronto llegue
- Si el servidor está saturado, el backpressure de gRPC maneja la presión de flujo
- Logging de latencia por frame para detectar bottlenecks

### T2.4 — Hot-Swap Thread-Safe

```python
class ModelManager:
    def hot_swap(self, new_config: ModelConfig) -> None:
        """
        1. Cargar nuevo modelo en background (sin interrumpir el activo)
        2. Adquirir write lock
        3. Hacer warmup del nuevo modelo
        4. Reemplazar referencia atómica
        5. Liberar memoria del modelo anterior
        6. Liberar write lock
        """
        with self._swap_lock:
            new_detector = self._load_detector(new_config)
            new_detector.warmup()
            self._active_detector = new_detector
            # El garbage collector libera el anterior
```

**Endpoint HTTP para hot-swap:**
```
POST /hot-swap
Body: {"model_name": "yolo11s", "backend": "onnx", "path": "/models/yolo11s.onnx"}
Response: {"status": "swapping"} → polling /health hasta {"status": "ready"}
```

### T2.5 — stream_predict() en Cliente

```python
class InferenceClient:
    def stream_predict(
        self,
        frame_generator: Iterator[np.ndarray]
    ) -> Iterator[sv.Detections]:
        """
        Genera InferenceRequests desde el frame_generator y
        hace yield de sv.Detections conforme llegan las respuestas.

        Uso típico:
            cap = cv2.VideoCapture(0)
            def frames():
                while True:
                    ret, frame = cap.read()
                    if ret: yield frame

            for detections in client.stream_predict(frames()):
                # procesar detecciones en tiempo real
                annotate(frame, detections)
        """
        def request_generator():
            for frame in frame_generator:
                resized = cv2.resize(frame, (self._input_width, self._input_height))
                _, encoded = cv2.imencode('.jpg', resized,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                yield detections_pb2.InferenceRequest(
                    image_data=encoded.tobytes(),
                    width=self._input_width,
                    height=self._input_height
                )

        for response in self._stub.StreamPredict(request_generator()):
            yield self._deserialize_response(response)
```

---

## Tests de Fase 2

### `test_rfdetr_detector.py`
- [ ] `preprocess()` aplica normalización ImageNet correctamente
- [ ] `postprocess()` escala coordenadas relativas → absolutas correctamente
- [ ] `postprocess()` filtra detecciones bajo `confidence_threshold`
- [ ] `predict()` retorna `sv.Detections` (usando RF-DETR small)
- [ ] `warmup()` completa sin errores

### `test_tensorrt_backend.py`
- [ ] `TensorRTBackend.is_available()` retorna `False` cuando no hay CUDA (no lanza excepción)
- [ ] Tests de inferencia marcados como `@pytest.mark.skipif(not cuda_available, reason="Requires CUDA")`

### `test_streaming.py`
- [ ] `StreamPredict` procesa 30 frames consecutivos sin errores
- [ ] Latencia por frame no degrada significativamente entre frame 1 y frame 30
- [ ] Stream se cierra limpiamente cuando el generator se agota
- [ ] `client.stream_predict()` retorna un iterator de `sv.Detections`

### `test_hot_swap.py`
- [ ] Hot-swap con el mismo tipo de modelo funciona
- [ ] Hot-swap de YOLO11 → RF-DETR funciona
- [ ] Las requests en vuelo durante hot-swap no fallan (usan el modelo anterior)
- [ ] `/hot-swap` endpoint retorna 200 y luego `/health` vuelve a `ready`

### `test_integration.py`
- [ ] Cliente + servidor YOLO11 end-to-end con imagen real
- [ ] Cliente + servidor RF-DETR end-to-end con imagen real
- [ ] Streaming 10 frames de video real retorna detecciones coherentes
- [ ] Hot-swap durante streaming no interrumpe el stream

---

## Definición de "Fase 2 Completa"

- [ ] RF-DETR y YOLO11 producen `sv.Detections` con el mismo contrato de API
- [ ] `stream_predict()` del cliente procesa video en tiempo real (>=15fps en CPU)
- [ ] Hot-swap cambia de modelo sin reiniciar Docker y sin fallar requests en curso
- [ ] TensorRT backend carga y produce resultados equivalentes a ONNX (en hardware CUDA)
- [ ] Todos los tests de Fase 2 pasan (con skipif apropiados para CUDA)
- [ ] JOURNAL.md actualizado con los cambios relevantes
