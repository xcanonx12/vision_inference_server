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

_Las siguientes entradas se agregarán conforme avance el desarrollo._
