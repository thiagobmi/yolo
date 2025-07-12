FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sshpass \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pré-carregar modelos YOLO
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('Modelo YOLO v8 nano baixado com sucesso')"

COPY . .

# Variável de ambiente para a porta (padrão: 8000)
ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "python3 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
