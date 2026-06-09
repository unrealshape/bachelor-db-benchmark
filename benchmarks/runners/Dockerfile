# Schlankes In-Cluster-Mess-Image: nur Query-Loop, kein Embedding (kein torch),
# kein pinecone. Wird per `k3d image import` in den Cluster geladen.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "numpy>=1.26" "pyarrow>=15" "weaviate-client>=4.7" "psycopg[binary]>=3.2"

WORKDIR /app
# Build-Context = benchmarks/runners (klein).
COPY . /app/benchmarks/runners

ENV PYTHONPATH=/app/benchmarks/runners
ENV BENCH_IN_CLUSTER=1
ENTRYPOINT ["python", "/app/benchmarks/runners/measure.py"]
