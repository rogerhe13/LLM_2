FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY . .


RUN pip install --no-cache-dir -r requirements.txt


CMD ["python3", "unit_test.py"]