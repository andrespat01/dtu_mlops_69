FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY app/frontend.py /app/frontend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt



# ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]

# Expose the specified port
EXPOSE $PORT

# Use a shell to resolve the environment variable for the port
ENTRYPOINT ["/bin/sh", "-c", "streamlit run frontend.py --server.port $PORT --server.address=0.0.0.0"]