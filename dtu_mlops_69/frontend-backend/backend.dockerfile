FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Set environment variable to point to the credentials file
ENV GOOGLE_APPLICATION_CREDENTIALS="gcloud-key.json"

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt
COPY app/backend.py /app/backend.py
COPY gcloud-key.json /app/gcloud-key.json


RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt

EXPOSE $PORT
CMD exec uvicorn backend:app --port $PORT --host 0.0.0.0 --workers 1
