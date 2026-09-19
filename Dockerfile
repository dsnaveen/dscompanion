FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by xgboost/lightgbm.
# default-jre-headless: PySpark (databricks extra) requires a JRE at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[all]"

CMD ["bash"]
