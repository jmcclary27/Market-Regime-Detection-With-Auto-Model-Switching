FROM apache/airflow:2.9.3-python3.11

USER root

# System deps if you need them, keep minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Airflow-side python deps (for DAG parsing, dotenv, etc.)
COPY docker/airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Install your project deps so tasks can run your entrypoints
# Assumes your repo has requirements.txt at root.
COPY requirements.txt /opt/project/requirements.txt
RUN pip install --no-cache-dir -r /opt/project/requirements.txt

# Copy your project code into the image
# If you prefer mounting instead, you can remove these COPY lines and mount the repo in compose.
COPY src /opt/project/src
COPY pyproject.toml /opt/project/pyproject.toml
COPY README.md /opt/project/README.md

ENV PYTHONPATH="/opt/project"
WORKDIR /opt/project
