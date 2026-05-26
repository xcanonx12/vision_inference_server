FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies and compile proto
COPY server/requirements.txt .
RUN pip install --no-cache-dir grpcio-tools==1.78.0

COPY server/proto/ server/proto/
RUN mkdir -p server/generated && \
    python -m grpc_tools.protoc \
        -I./server/proto \
        --python_out=./server/generated \
        --grpc_python_out=./server/generated \
        ./server/proto/detections.proto && \
    # Fix relative import in generated grpc file
    sed -i 's/^import detections_pb2/from . import detections_pb2/' \
        server/generated/detections_pb2_grpc.py && \
    touch server/generated/__init__.py

# ---

FROM python:3.11-slim-bookworm AS production

WORKDIR /app

# Install system dependencies for OpenCV + curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy generated proto files from builder
COPY --from=builder /app/server/generated/ server/generated/

# Copy application code
COPY server/ server/
COPY client/ client/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 50051 8080

CMD ["python", "-m", "server.main"]
