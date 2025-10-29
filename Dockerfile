FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

COPY . /app

CMD ["python", "-c", "print('Image ready. Override CMD in docker-compose to run generate_triples.py')"]
