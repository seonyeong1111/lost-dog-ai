FROM python:3.12-slim

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["python", "app.py"]
