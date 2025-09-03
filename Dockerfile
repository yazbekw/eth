FROM python:3.9-slim-buster

WORKDIR /app

# تثبيت dependencies النظام المطلوبة
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "eth.py"]
