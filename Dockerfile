FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501