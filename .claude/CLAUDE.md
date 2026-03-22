# CLAUDE.md — Guía de Trabajo para SimpleInference

Este archivo describe las convenciones, reglas y procesos que deben seguirse durante el desarrollo de SimpleInference. Aplica a cualquier agente de IA o colaborador que trabaje en este proyecto.

---

## Contexto del Proyecto

SimpleInference es un servidor de inferencia de visión artificial con las siguientes características clave:
- **Protocolo:** gRPC (streaming bidireccional)
- **Caso de uso primario:** Tiempo real — cámaras en vivo, drones, edge devices
- **Salida universal:** `sv.Detections` de Roboflow Supervision
- **Configuración:** YAML
- **Deploy:** Docker (solo el servidor)
- **Cliente:** Archivo `.py` único portable

Antes de implementar cualquier cosa, consultar `ARCHITECTURE.md` para entender el diseño y `PLAN.md` para el contexto de decisiones tomadas.

---

## Reglas Generales

### 1. No hardcodear valores
Cualquier valor configurable (dimensiones del modelo, umbrales, puertos, rutas) debe venir del `config.yaml`. No usar magic numbers en el código.

### 2. Salida siempre es `sv.Detections`
Todos los detectores deben retornar `sv.Detections`. Nunca retornar diccionarios, listas o estructuras propias. Esto es innegociable.

### 3. Tipado estricto
Usar type hints en todas las funciones públicas. Usar `dataclass` o `pydantic` para configuración, nunca diccionarios sin tipar.

### 4. Manejo de errores en gRPC
Nunca dejar que una excepción Python propague al cliente gRPC sin capturarla. Siempre usar `context.abort(StatusCode.X, "mensaje descriptivo")`.

### 5. Logging estructurado
Usar el módulo estándar `logging`. Formato: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`. No usar `print()` en código de producción.

### 6. El cliente es portable
`inference_client.py` debe funcionar copiando el archivo solo. No puede importar módulos internos del servidor. Sus únicas dependencias son: `grpcio`, `opencv-python`, `numpy`, `supervision` y los archivos `_pb2.py` generados.

---

## Proceso de Desarrollo por Fase

### Al iniciar una fase:
1. Leer el archivo `PHASEX.md` correspondiente
2. Verificar que la fase anterior está marcada como completa
3. Crear branch `phase/X-nombre` desde `main`

### Durante el desarrollo:
1. Implementar una tarea a la vez (T1.1, T1.2, etc.)
2. Escribir tests antes o junto con la implementación (no después)
3. Registrar en `JOURNAL.md` cualquier cambio significativo (ver sección JOURNAL más abajo)

### Al terminar una fase:
1. **Ejecutar todos los unit tests:** `pytest server/tests/ client/tests/ -v`
2. Verificar cobertura mínima del 80%: `pytest --cov=. --cov-report=term-missing`
3. Marcar todos los checkboxes del archivo `PHASEХ.md`
4. Actualizar `JOURNAL.md` con el resumen de la fase
5. Hacer merge a `main`

---

## Tests

### Reglas de Testing

- **Framework:** `pytest` exclusivamente
- **Modelos reales en tests:** usar siempre las versiones más ligeras disponibles:
  - YOLO11: `yolo11n` (nano)
  - RF-DETR: versión small/base más ligera disponible
- **Tests que requieren GPU:** marcar con `@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")`
- **Tests de carga:** marcar con `@pytest.mark.slow` y excluir del CI por defecto
- **No tests frágiles:** no usar `time.sleep()` en tests; usar mocks o esperas basadas en eventos

### Estructura de Tests

```
server/tests/
├── conftest.py           # fixtures compartidos (server mock, sample images, etc.)
├── test_config_loader.py
├── test_device_manager.py
├── test_backends.py
├── test_detectors.py
├── test_grpc_service.py
└── test_http_service.py

client/tests/
├── conftest.py
├── test_client.py
└── test_integration.py   # requiere servidor corriendo
```

### Ejecutar Tests

```bash
# Tests unitarios (rápidos, sin GPU requerida)
pytest server/tests/ client/tests/ -v -m "not slow"

# Tests de integración (requiere servidor corriendo)
pytest client/tests/test_integration.py -v

# Tests de carga (slow)
pytest -m slow -v

# Con cobertura
pytest --cov=server --cov=client --cov-report=html
```

### Datos Sintéticos para Tests

Usar el helper de datos sintéticos para tests que no requieran imágenes reales:

```python
# conftest.py
@pytest.fixture
def synthetic_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def synthetic_detections():
    return sv.Detections(
        xyxy=np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 1], dtype=np.int32),
    )
```

---

## JOURNAL.md — Qué Registrar

Registrar en `JOURNAL.md` únicamente cambios con impacto funcional:

### ✅ SÍ registrar:
- Decisiones de arquitectura y su justificación
- Cambios en el contrato de API (proto, endpoints HTTP)
- Cambios en el formato de configuración YAML
- Bugs importantes y su solución
- Benchmarks y resultados de performance
- Cambios de dependencias (agregar/eliminar paquetes)
- Resultados de tests de carga
- Problemas de compatibilidad entre backends

### ❌ NO registrar:
- Correcciones tipográficas
- Refactoring interno sin cambio de comportamiento
- Ajuste de comentarios o docstrings
- Cambios de formato/estilo (black, isort)
- Actualización de versiones de dependencias sin cambio de comportamiento

### Formato de entrada en JOURNAL:
```markdown
## [FASE X] YYYY-MM-DD — Título descriptivo

**Tipo:** feat | fix | perf | breaking | decision

**Descripción:**
Qué cambió y por qué.

**Impacto:**
Qué afecta este cambio (API, config, performance, etc.)

**Notas:**
Información adicional relevante (benchmarks, trade-offs, etc.)
```

---

## Dependencias y Entorno

### Versiones fijadas (agregar al requirements.txt con versión exacta):
```
grpcio==X.X.X
grpcio-tools==X.X.X
supervision==X.X.X
ultralytics==X.X.X
rfdetr==X.X.X          # fijar versión por ser paquete reciente
fastapi==X.X.X
uvicorn==X.X.X
pyyaml==X.X.X
opencv-python==X.X.X
numpy==X.X.X
```

### Variables de entorno (nunca en código, siempre en `.env`):
```
ROBOFLOW_API_KEY=...    # requerido solo si source=roboflow en config.yaml
```

### Compilar proto:
```bash
python -m grpc_tools.protoc \
  -I./proto \
  --python_out=./generated \
  --grpc_python_out=./generated \
  ./proto/detections.proto
```

---

## Checklist Final de Fase

Al completar cada fase, verificar:

- [ ] Todos los tests pasan: `pytest -v -m "not slow"`
- [ ] Cobertura >= 80%: `pytest --cov --cov-fail-under=80`
- [ ] No hay `print()` en código de producción
- [ ] No hay valores hardcodeados fuera de `config.yaml`
- [ ] `JOURNAL.md` tiene al menos una entrada con los cambios relevantes de la fase
- [ ] El archivo `PHASEХ.md` tiene todos los checkboxes marcados
- [ ] `docker-compose up` levanta sin errores (Fase 1 en adelante)
