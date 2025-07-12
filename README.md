# NUVYOLO - API de Detecção de Objetos

API para detecção e rastreamento de objetos em streams de vídeo utilizando YOLO.

## Estrutura do Projeto

```
app/
├── __init__.py
├── main.py                 
├── config/                 # Configurações da aplicação
├── api/                    # Definições da API
│   ├── models/             # Modelos Pydantic
│   ├── routes/             # Endpoints da API
├── core/                   # Lógica de negócio
├── external/               # Integração com APIs externas
├── utils/                  # Funções utilitárias
```

## Requisitos

- Python 3.10+
- OpenCV
- FastAPI
- YOLO (Ultralytics)
- sshpass
- nvidia-container-toolkit
- Outros requisitos em `requirements.txt`

## Configuração

1. Clonar o repositório:
   ```bash
   git clone <repositório>
   cd nuvyolo
   ```

2. Criar um ambiente virtual e instalar as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configurar as variáveis de ambiente (`.env`).

4. Configurar as especificações do NUV em `sample_specifications.json`.

5. Baixe os modelos YOLO (YOLOv8) (Opcinal, YOLO baixa durante a execução)`:
   ```bash
   # Baixe os modelos manualmente ou use o script:
   # python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

6. Instalar o nvidia-container-toolkit:
   ```
   # Configurar o repositório
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Atualizar lista de pacotes
   sudo apt-get update

   # Instalar nvidia-container-toolkit
   sudo apt-get install -y nvidia-container-toolkit

   # Reiniciar o Docker para aplicar as alterações
   sudo systemctl restart docker
   ```

7. Configurar o Docker para usar o runtime NVIDIA
   ```
   # Configure o Docker para usar o NVIDIA Container Toolkit
   sudo nvidia-ctk runtime configure --runtime=docker

   # Reinicie o Docker para aplicar as alterações
   sudo systemctl restart docker
   ```

8. Instale o sshpass
   ```
   sudo apt install sshpass
   ```
   
> Verficiar se o runtime está disponivel:`docker info | grep -i runtime`


## Executando a Aplicação

### Desenvolvimento

```bash
uvicorn app.main:app --reload
```

### Produção

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Com Docker

```bash
docker compose up -d
```

## Executando o site demo para visualização dos eventos

```bash
cd ./event_viewer_demo 
python main.py
```

## API Endpoints

- `GET /`: Informações sobre a API
- `POST /monitor`: Iniciar monitoramento em uma câmera
- `POST /stop/{camera_id}`: Parar monitoramento em uma câmera
- `POST /stop/all`: Parar monitoramento em todas as câmeras
- `GET /monitored`: Listar câmeras monitoradas

## Exemplo de Uso

### Iniciar Monitoramento

```bash
curl -X POST "http://localhost:8000/monitor" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": 51
    "device": "cpu",
    "detection_model_path": "models/yolov8n.pt",
    "classes": ["person", "car", "truck"],
    "tracker_model": "bytetrack.yaml",
    "frames_per_second": 5,
    "frames_before_disappearance": 30,
    "confidence_threshold": 0.25,
    "iou": 0.45
  }'
```

### Parar Monitoramento

```bash
curl -X POST "http://localhost:8000/stop/51"
```

### Verificar Câmeras Monitoradas

```bash
curl "http://localhost:8000/monitored"
```

## Documentação da API

A documentação automática da API está disponível em:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
