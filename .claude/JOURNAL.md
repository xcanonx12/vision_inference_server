# JOURNAL — SimpleInference

Registro de cambios con impacto funcional en el proyecto.  
Ver `CLAUDE.md` para las reglas sobre qué registrar y el formato esperado.

---

## [INICIO] 2025-03-21 — Definición del proyecto y documentación inicial

**Tipo:** decision

**Descripción:**
Creación del conjunto inicial de documentación del proyecto: `ARCHITECTURE.md`, `PLAN.md`, `PHASE1.md`, `PHASE2.md`, `PHASE3.md`, `CLAUDE.md`, `REFERENCES.md`.

Se estableció la arquitectura completa basada en los siguientes principios y decisiones:

**Decisiones arquitectónicas clave:**

1. **Un modelo activo a la vez (hot-swap)** — Se descartó concurrent execution en Fase 1 y 2 para mantener la arquitectura simple y optimizada para el caso de uso primario (tiempo real, edge). Concurrent execution se implementará en Fase 3.

2. **Cliente como archivo `.py` único portable** — Se descartó publicar en PyPI para Fase 1/2/3. La portabilidad máxima (copy-paste) tiene más valor que la distribución formal dado el contexto de uso.

3. **Espera activa en cliente hasta servidor listo** — Fundamental para Docker Compose donde el servidor puede tardar 10–30s en cargar el modelo y completar warmup antes de estar listo.

4. **`sv.Detections` como contrato universal de salida** — Todos los detectores retornan esta estructura independientemente del modelo o backend. Elimina la necesidad de adapters en el código consumidor.

5. **FastAPI para endpoints HTTP en paralelo a gRPC** — `/health`, `/config`, `/metrics` son más ergonómicos como REST que como gRPC para consumo desde herramientas externas (curl, Grafana, scripts de monitoring).

6. **YAML para configuración** — Soporta comentarios, estructura jerárquica natural para configuración de modelos con versiones, dimensiones y backends.

7. **Carga de modelos: local primero, descarga si no existe** — Balance entre reproducibilidad (modelos en disco) y conveniencia (descarga automática desde Roboflow Hub).

**Protobuffer base (del prototipo existente):**
Se toma como punto de partida el `detections.proto` del prototipo, añadiendo:
- `StreamPredict` RPC para streaming bidireccional
- `GetServerConfig` RPC para que el cliente sincronice configuración
- `ServerConfigResponse` message

**Stack tecnológico definido:**
Python + gRPC + Supervision + OpenCV + FastAPI + Docker. Se añade `ultralytics`, `rfdetr`, `onnxruntime`, `pyyaml`, `pytest` al stack base.

**Impacto:**
Toda la documentación base está creada. El siguiente paso es implementar Fase 1, comenzando por T1.1 (proto actualizado) y T1.2 (config loader).

---

## [FASE 1] 2026-03-21 — Implementación completa de fundamentos

**Tipo:** feat

**Descripción:**
Implementación end-to-end de la Fase 1: servidor gRPC + FastAPI, YOLO11 detector con backend PyTorch, cliente portable, y configuración Docker.

**Componentes implementados:**
- `detections.proto` con Predict, StreamPredict (stub), GetServerConfig RPCs
- Config loader con dataclasses tipados y validación YAML
- Device manager con detección CPU/CUDA y cadena de fallback
- Backend layer (BaseBackend ABC, PyTorchBackend, ONNXBackend) usando ultralytics
- BaseDetector ABC + YOLO11Detector con salida `sv.Detections`
- ModelManager con registros extensibles de backends y detectores
- gRPC InferenceServicer con manejo de errores via context.abort
- FastAPI con /health (ready/loading) y /config
- Main entrypoint con gRPC + FastAPI concurrentes y graceful shutdown
- InferenceClient portable con connect (espera activa), fetch_config, predict
- Dockerfile multi-stage + docker-compose.yml

**Decisión: Backend retorna objetos nativos, no np.array wrapper**
Los backends retornan listas de ultralytics Results directamente en vez de envolverlos en `np.array(results, dtype=object)`, ya que esto perdía la interfaz del objeto Results necesaria para `sv.Detections.from_ultralytics()`.

**Decisión: ModelConfig soporta modelos custom/fine-tuned**
`class_names` es una lista opcional (default vacía). `name` es freeform. `path` acepta cualquier archivo compatible. No se valida contra nombres de modelos pretrained.

**Impacto:**
Sistema funcional end-to-end. El servidor puede cargar YOLO11n, recibir imágenes vía gRPC, y retornar `sv.Detections`. El cliente puede conectarse, sincronizar config, y hacer inferencia.

**Notas:**
- 50 tests, todos pasando
- Cobertura: 84% (sobre el mínimo de 80%)
- StreamPredict definido en proto pero stubbed como UNIMPLEMENTED (Fase 2)
- Hot-swap preparado en ModelManager pero no implementado (Fase 2)

---

_Las siguientes entradas se agregarán conforme avance el desarrollo._
