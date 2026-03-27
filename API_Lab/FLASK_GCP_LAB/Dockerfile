FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model
COPY src ./src

CMD ["sh", "-c", "cd src && uvicorn main:app --host 0.0.0.0 --port $PORT"]
