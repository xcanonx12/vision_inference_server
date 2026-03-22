# SimpleInference — Plan de Desarrollo

> **Stack:** Python · gRPC · Supervision · OpenCV · FastAPI · Docker  
> **Caso de uso:** Inferencia en tiempo real (cámaras, drones, edge)  
> **Distribución del cliente:** archivo `.py` portable (copy-paste a cualquier proyecto)

---

## Resumen de Fases

| Fase | Nombre | Enfoque | Entregable Principal |
|------|--------|---------|---------------------|
| **1** | Fundamentos | Abstracción + YOLO11 funcional | Servidor gRPC + cliente operativos end-to-end |
| **2** | Completitud | RF-DETR + streaming + hot-swap | Sistema completo con todos los backends |
| **3** | Producción | Performance + observabilidad | Sistema production-ready con métricas y concurrencia |

---

## Stack Tecnológico Completo

### Servidor
| Componente | Librería | Propósito |
|---|---|---|
| Inferencia gRPC | `grpcio`, `grpcio-tools` | Protocolo de comunicación |
| HTTP API | `fastapi`, `uvicorn` | `/health`, `/config`, `/metrics` |
| Detección YOLO | `ultralytics` | YOLO11 detector |
| Detección DETR | `rfdetr` | RF-DETR detector |
| Backend ONNX | `onnxruntime` / `onnxruntime-gpu` | Inferencia ONNX |
| Backend TensorRT | `tensorrt`, `pycuda` | Inferencia TRT (CUDA only) |
| Estandarización | `supervision` | Salida unificada `sv.Detections` |
| Procesamiento | `opencv-python`, `numpy` | Pre/post processing |
| Configuración | `pyyaml` | YAML config loader |
| Descarga modelos | `roboflow` (inference SDK) | Auto-download desde Roboflow Hub |
| Contenedor | `docker`, `docker-compose` | Deploy del servidor |

### Cliente
| Componente | Librería | Propósito |
|---|---|---|
| gRPC | `grpcio` | Comunicación con servidor |
| Imagen | `opencv-python`, `numpy` | Resize + encode |
| Salida | `supervision` | Deserializar a `sv.Detections` |

---

## Criterios de Éxito por Fase

### Fase 1 ✓ cuando:
- [ ] El servidor arranca con `docker-compose up` sin errores
- [ ] El cliente se conecta y recibe `sv.Detections` reales de YOLO11n
- [ ] `/health` y `/config` responden correctamente
- [ ] El warmup completa antes de servir tráfico
- [ ] Tests unitarios pasan al 100%

### Fase 2 ✓ cuando:
- [ ] RF-DETR funciona igual que YOLO11 (mismo contrato de API)
- [ ] Streaming bidireccional procesa video en tiempo real sin drops
- [ ] Hot-swap cambia de modelo sin reiniciar el servidor
- [ ] Los 3 backends (PyTorch, ONNX, TensorRT) producen resultados equivalentes
- [ ] Tests de integración pasan al 100%

### Fase 3 ✓ cuando:
- [ ] `/metrics` expone latencia P50/P95/P99, throughput y conteo de errores
- [ ] Dynamic batching mejora throughput >= 20% vs batch_size=1 en benchmark
- [ ] Concurrent model execution funciona (2 modelos simultáneos)
- [ ] Tests de carga pasan: 30fps sostenido por >= 60 segundos
- [ ] Documentación de usuario final completa

---

## Convenciones del Proyecto

### Commits
```
[PHASE1] feat: implement BaseDetector abstract class
[PHASE2] fix: ONNX backend wrong output shape for RF-DETR
[PHASE3] perf: dynamic batching reduces latency 30ms→12ms
```

### Tests
- Framework: `pytest`
- Modelos en tests: versiones nano/tiny (YOLO11n, RF-DETR small) para velocidad
- Mocks: usar datos sintéticos para tests que no requieran GPU
- Cobertura mínima por fase: **80%** en código nuevo

### Configuración
- Todo lo configurable va en `config.yaml`, nunca hardcodeado
- Variables de entorno solo para secretos (API keys de Roboflow)
- El servidor debe arrancar con valores por defecto si no hay config.yaml

### Branching sugerido
```
main
├── phase/1-foundations
├── phase/2-completeness
└── phase/3-production
```

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| TensorRT requiere hardware específico para tests | Alta | Tests de TensorRT marcados como `@pytest.mark.skipif(no_cuda)` |
| RF-DETR API cambia (paquete reciente) | Media | Pin versión exacta en requirements.txt |
| Latencia gRPC overhead en edge devices | Baja | Benchmark comparativo en Fase 3, ajustar buffer sizes |
| Descarga de modelos Roboflow en CI/CD | Media | Cache de modelos en tests, mock del downloader |

---

## Decisiones de Diseño Tomadas

| Decisión | Alternativa Descartada | Razón |
|----------|----------------------|-------|
| Un modelo activo a la vez (hot-swap) | Múltiples modelos paralelos | Simplicidad + suficiente para edge en tiempo real |
| Cliente como archivo `.py` único | PyPI package | Portabilidad máxima, cero fricción de instalación |
| FastAPI para HTTP endpoints | Solo gRPC | `/metrics` y `/config` son más fáciles de consumir desde cualquier herramienta |
| `sv.Detections` como salida universal | Formato propio | Supervision es estándar en el ecosistema Roboflow, tiene métodos de post-proceso built-in |
| Espera activa en cliente | Fail-fast | Crítico para Docker Compose donde servidor puede tardar en estar listo |
| YAML para configuración | ENV vars / JSON | Más legible, soporta comentarios, ideal para estructura jerárquica de modelo |
