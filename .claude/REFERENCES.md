# Referencias

Recursos técnicos y documentación relevante para SimpleInference.

---

## Frameworks de Inferencia

### Supervision (Roboflow)
- Documentación oficial: https://supervision.roboflow.com/
- API Reference `sv.Detections`: https://supervision.roboflow.com/latest/detection/core/
- GitHub: https://github.com/roboflow/supervision
- Métodos de post-procesamiento disponibles: `filter_by_confidence()`, `with_nms()`, `merge()`, `__len__()`

### Inference (Roboflow Python SDK)
- Documentación: https://inference.roboflow.com/
- GitHub: https://github.com/roboflow/inference
- Uso principal en este proyecto: descarga automática de modelos desde Roboflow Hub

### Ultralytics YOLO11
- Documentación: https://docs.ultralytics.com/
- YOLO11 announcement: https://docs.ultralytics.com/models/yolo11/
- Export a ONNX: https://docs.ultralytics.com/modes/export/
- Modelos disponibles: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

### RF-DETR
- Paper: "RF-DETR: Real-Time Detection Transformer" (Roboflow, Marzo 2025)
- GitHub: https://github.com/roboflow/rf-detr
- Blog post: https://blog.roboflow.com/rf-detr/
- Nota: primer detector transformer en tiempo real con 60+ mAP, sin NMS, Apache 2.0

---

## gRPC

### Documentación Core
- gRPC Python quickstart: https://grpc.io/docs/languages/python/quickstart/
- Protocol Buffers (proto3): https://protobuf.dev/programming-guides/proto3/
- Streaming gRPC: https://grpc.io/docs/what-is-grpc/core-concepts/#server-streaming-rpc
- gRPC Status Codes: https://grpc.github.io/grpc/core/md_doc_statuscodes.html

### Patrones Relevantes
- Bidirectional streaming: https://grpc.io/docs/what-is-grpc/core-concepts/#bidirectional-streaming-rpc
- Interceptors (para métricas): https://grpc.github.io/grpc/python/grpc.html#interceptor
- Health checking protocol: https://github.com/grpc/grpc/blob/master/doc/health-checking.md

---

## Backends de Inferencia

### ONNX Runtime
- Documentación Python: https://onnxruntime.ai/docs/api/python/
- Providers disponibles (CPU, CUDA, TensorRT): https://onnxruntime.ai/docs/execution-providers/
- Optimización de performance: https://onnxruntime.ai/docs/performance/
- Export PyTorch → ONNX: https://pytorch.org/docs/stable/onnx.html

### TensorRT
- Documentación NVIDIA: https://docs.nvidia.com/deeplearning/tensorrt/
- Python API: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/
- Guía de optimización: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- Precisión FP16/INT8: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#working-with-mixed-precision

### PyTorch
- Documentación: https://pytorch.org/docs/stable/
- torch.cuda: https://pytorch.org/docs/stable/cuda.html
- Model optimization (torch.compile): https://pytorch.org/docs/stable/torch.compile.html

---

## FastAPI
- Documentación: https://fastapi.tiangolo.com/
- Background tasks: https://fastapi.tiangolo.com/tutorial/background-tasks/
- Correr junto a gRPC (threading): https://fastapi.tiangolo.com/advanced/run-in-threadpool/

---

## Docker

### Referencias
- Dockerfile best practices: https://docs.docker.com/develop/dev-best-practices/
- Multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- Docker Compose health checks: https://docs.docker.com/compose/compose-file/05-services/#healthcheck
- NVIDIA Container Toolkit (GPU en Docker): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

### Base Images Recomendadas
- CPU: `python:3.11-slim-bookworm`
- CUDA + CUDNN: `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`
- CUDA dev (para compilar TensorRT): `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04`

---

## OpenCV

- Documentación Python: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- `cv2.imencode` (JPEG compression): https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
- `cv2.resize` interpolation flags: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html

---

## Testing

- pytest documentación: https://docs.pytest.org/
- pytest-cov (cobertura): https://pytest-cov.readthedocs.io/
- Testing gRPC services en Python: https://grpc.io/docs/languages/python/testing/

---

## Proyectos de Referencia (Arquitectura)

Estos proyectos sirvieron de inspiración para las decisiones de diseño:

| Proyecto | Qué aprendimos |
|---|---|
| **NVIDIA Triton Inference Server** | Estructura de backends, dynamic batching, concurrent model execution |
| **Roboflow Inference** | Simplicidad de API, estandarización con `sv.Detections`, gestión de modelos |
| **Ultralytics** | Ergonomía del usuario, YAML config, warmup patterns |
| **BentoML** | Serialización eficiente, runner patterns |

---

## Notas Importantes

### RF-DETR vs YOLO11 — Diferencias clave de implementación
- RF-DETR **no requiere NMS** (el modelo es un transformer que ya produce detecciones únicas)
- RF-DETR usa normalización ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- YOLO11 normaliza dividiendo entre 255.0 sin media/std adicional
- Las coordenadas de salida de RF-DETR son **relativas** [0,1]; YOLO11 retorna coordenadas absolutas en píxeles

### ONNX Runtime Providers — Orden de prioridad
```python
# Orden recomendado para selección automática de provider
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# onnxruntime selecciona el primero disponible automáticamente
session = ort.InferenceSession(model_path, providers=providers)
```
