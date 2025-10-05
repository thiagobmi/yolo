from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional
import uvicorn
import os
import uuid
import base64
from datetime import datetime
import logging
from PIL import Image, ImageDraw

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar diretórios necessários
os.system("rm -f static/images/*")
os.system("rm -f static/cropped_images/*")
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/cropped_images", exist_ok=True)
os.makedirs("static/css", exist_ok=True)

# Inicializar FastAPI
app = FastAPI(title="Visualizador de Eventos")

# Configurar templates e arquivos estáticos
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model para eventos recebidos
class EventReceive(BaseModel):
    camera_id: int
    start: str
    end: str
    event_type: str
    tag: str  # classe do objeto (carro, pessoa, etc)
    coord_initial: Tuple[int, int]
    coord_end: Tuple[int, int]
    print: str  # imagem codificada em hexadecimal

# Armazenamento em memória para eventos (em produção, use um banco de dados)
events_storage = {}

def crop_and_save_object(image_bytes, coord_initial, coord_end, event_id):
    """
    Recorta apenas a região da bounding box e salva em cropped_images.
    
    Args:
        image_bytes: Bytes da imagem original
        coord_initial: Coordenada inicial (x, y)
        coord_end: Coordenada final (x, y)
        event_id: ID único do evento
    
    Returns:
        Caminho do arquivo da imagem recortada
    """
    try:
        import io
        
        # Abrir a imagem a partir dos bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Extrair coordenadas
        x1, y1 = coord_initial
        x2, y2 = coord_end
        
        # Garantir que as coordenadas estejam dentro dos limites da imagem
        width, height = img.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Recortar a região da bounding box
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Salvar a imagem recortada
        cropped_path = f"static/cropped_images/{event_id}_cropped.jpg"
        cropped_img.save(cropped_path, "JPEG", quality=95)
        
        logger.info(f"Imagem recortada salva: {cropped_path}")
        return cropped_path
        
    except Exception as e:
        logger.error(f"Erro ao recortar imagem: {e}")
        return None

def draw_bounding_box(image_bytes, coord_initial, coord_end, tag):
    """
    Desenha uma bounding box na imagem e adiciona a etiqueta
    Args:
        image_bytes: Bytes da imagem
        coord_initial: Coordenada inicial (x, y)
        coord_end: Coordenada final (x, y)
        tag: Nome do objeto detectado
    """
    try:
        # Abrir a imagem a partir dos bytes
        import io
        img = Image.open(io.BytesIO(image_bytes))
        
        # Preparar o objeto de desenho
        draw = ImageDraw.Draw(img)
        
        # Desenhar o retângulo
        # Cores para diferentes tipos de objetos
        color_map = {
            "person": (255, 0, 0),  # Vermelho
            "car": (0, 255, 0),   # Verde
            "truck": (0, 0, 255), # Azul
            "train": (255, 255, 0),  # Amarelo
            "bike": (255, 0, 255), # Magenta
        }
        
        # Cor padrão se a tag não estiver no mapa
        color = color_map.get(tag.lower(), (255, 165, 0))  # Laranja como padrão
        
        # Desenhar o retângulo com espessura 3
        draw.rectangle([coord_initial, coord_end], outline=color, width=3)
        
        # Adicionar texto com a tag
        text_position = (coord_initial[0], coord_initial[1] - 20)
        draw.text(text_position, tag, fill=color)
        
        # Converter a imagem de volta para bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Erro ao desenhar bounding box: {e}")
        # Retornar a imagem original se houver erro
        return image_bytes

@app.post("/events/receive")
async def receive_event(event: EventReceive):
    """
    Endpoint para receber eventos do sistema de detecção de objetos.
    """
    try:
        # Gerar ID único para o evento
        event_id = str(uuid.uuid4())
        
        # Converter hexadecimal para bytes
        image_bytes = bytes.fromhex(event.print)
        
        # 1. Salvar imagem recortada (apenas o objeto) em cropped_images/
        cropped_path = crop_and_save_object(
            image_bytes,
            event.coord_initial,
            event.coord_end,
            event_id
        )
        
        # 2. Desenhar a bounding box na imagem completa
        image_with_bbox = draw_bounding_box(
            image_bytes, 
            event.coord_initial, 
            event.coord_end, 
            event.tag
        )
        
        # 3. Salvar a imagem completa com a bounding box em images/
        image_path = f"static/images/{event_id}.jpg"
        with open(image_path, "wb") as img_file:
            img_file.write(image_with_bbox)
        
        # Criar timestamp legível
        try:
            start_dt = datetime.fromisoformat(event.start)
            end_dt = datetime.fromisoformat(event.end)
            start_formatted = start_dt.strftime("%d/%m/%Y %H:%M:%S")
            end_formatted = end_dt.strftime("%d/%m/%Y %H:%M:%S")
            duration = (end_dt - start_dt).total_seconds()
        except ValueError:
            # Fallback se não conseguir parsear a data
            start_formatted = event.start
            end_formatted = event.end
            duration = 0
        
        # Armazenar evento com informações adicionais
        events_storage[event_id] = {
            "id": event_id,
            "camera_id": event.camera_id,
            "start": start_formatted,
            "end": end_formatted,
            "event_type": event.event_type,
            "tag": event.tag,
            "duration": f"{duration:.2f} segundos",
            "coord_initial": event.coord_initial,
            "coord_end": event.coord_end,
            "image_path": image_path,
            "cropped_path": cropped_path,  # Caminho da imagem recortada
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }
        
        logger.info(f"Evento recebido e armazenado com ID: {event_id}")
        logger.info(f"Imagem completa: {image_path}")
        logger.info(f"Imagem recortada: {cropped_path}")
        
        return {"status": "success", "event_id": event_id}
        
    except Exception as e:
        logger.error(f"Erro ao processar evento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Página inicial que mostra a lista de eventos.
    """
    # Obter todos os eventos e ordenar por timestamp de recebimento (mais recentes primeiro)
    events_list = list(events_storage.values())
    events_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "events": events_list}
    )

@app.get("/event/{event_id}", response_class=HTMLResponse)
async def view_event(request: Request, event_id: str):
    """
    Página de detalhes do evento.
    """
    if event_id not in events_storage:
        raise HTTPException(status_code=404, detail="Evento não encontrado")
    
    event = events_storage[event_id]
    
    return templates.TemplateResponse(
        "event_detail.html",
        {"request": request, "event": event}
    )

@app.get("/api/events")
async def list_events():
    """
    API para listar todos os eventos em formato JSON.
    """
    events_list = list(events_storage.values())
    events_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"events": events_list}

# Rota para limpar todos os eventos (útil para testes)
@app.post("/api/clear-events")
async def clear_events():
    """
    Limpa todos os eventos armazenados.
    """
    events_storage.clear()
    
    # Remover imagens completas
    for file in os.listdir("static/images"):
        if file.endswith(".jpg"):
            try:
                os.remove(os.path.join("static/images", file))
            except Exception as e:
                logger.error(f"Erro ao remover arquivo: {e}")
    
    # Remover imagens recortadas
    for file in os.listdir("static/cropped_images"):
        if file.endswith(".jpg"):
            try:
                os.remove(os.path.join("static/cropped_images", file))
            except Exception as e:
                logger.error(f"Erro ao remover arquivo recortado: {e}")
    
    return {"status": "success", "message": "Todos os eventos foram removidos"}

# Criar CSS para a aplicação
css_content = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    background-color: #f7f9fc;
    color: #333;
}

.container {
    width: 90%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2 {
    color: #2c3e50;
}

header {
    background-color: #34495e;
    color: white;
    padding: 1rem;
    margin-bottom: 2rem;
}

.events-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    grid-gap: 20px;
}

.event-card {
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.event-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.event-image {
    width: 100%;
    height: 200px;
    object-fit: cover;
}

.event-details {
    padding: 15px;
}

.event-title {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 10px;
    color: #2c3e50;
}

.event-info {
    font-size: 0.9rem;
    color: #7f8c8d;
}

.detail-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.detail-image {
    max-width: 100%;
    border-radius: 8px;
    margin-bottom: 20px;
}

.image-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.image-box {
    text-align: center;
}

.image-box h3 {
    margin-top: 0;
    margin-bottom: 10px;
    color: #2c3e50;
    font-size: 1rem;
}

.image-box img {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.detail-info {
    margin-bottom: 20px;
}

.detail-info dt {
    font-weight: bold;
    color: #2c3e50;
}

.detail-info dd {
    margin-left: 0;
    margin-bottom: 10px;
    color: #555;
}

.btn {
    display: inline-block;
    padding: 8px 16px;
    background-color: #3498db;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.9rem;
    transition: background-color 0.3s ease;
}

.btn:hover {
    background-color: #2980b9;
}

.no-events {
    text-align: center;
    padding: 40px;
    color: #7f8c8d;
    font-size: 1.2rem;
}

/* Responsividade */
@media (max-width: 768px) {
    .events-container {
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    }
    
    .image-comparison {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 480px) {
    .events-container {
        grid-template-columns: 1fr;
    }
}
"""

# Criar arquivos de templates
os.makedirs("templates", exist_ok=True)

# Template para página inicial
index_html = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizador de Eventos</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>Visualizador de Eventos</h1>
        </div>
    </header>
    
    <div class="container">
        {% if events %}
            <div class="events-container">
                {% for event in events %}
                <div class="event-card">
                    <img src="/{{ event.image_path }}" alt="Evento {{ event.id }}" class="event-image">
                    <div class="event-details">
                        <div class="event-title">{{ event.tag }} (Câmera {{ event.camera_id }})</div>
                        <div class="event-info">
                            <p>Detectado em: {{ event.timestamp }}</p>
                            <p>Duração: {{ event.duration }}</p>
                        </div>
                        <a href="/event/{{ event.id }}" class="btn">Ver detalhes</a>
                    </div>
                </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="no-events">
                <p>Nenhum evento detectado ainda.</p>
            </div>
        {% endif %}
    </div>
    
</body>
</html>
"""

# Template para página de detalhes do evento (ATUALIZADO com imagem recortada)
event_detail_html = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detalhes do Evento - {{ event.tag }}</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>Detalhes do Evento</h1>
        </div>
    </header>
    
    <div class="container">
        <div class="detail-container">
            <!-- Comparação de imagens: Completa vs Recortada -->
            <div class="image-comparison">
                <div class="image-box">
                    <h3>Imagem Completa com Bounding Box</h3>
                    <img src="/{{ event.image_path }}" alt="Imagem completa">
                </div>
                
                {% if event.cropped_path %}
                <div class="image-box">
                    <h3>Objeto Recortado</h3>
                    <img src="/{{ event.cropped_path }}" alt="Objeto recortado">
                </div>
                {% endif %}
            </div>
            
            <div class="detail-info">
                <dl>
                    <dt>ID do Evento:</dt>
                    <dd>{{ event.id }}</dd>
                    
                    <dt>Tipo de Objeto:</dt>
                    <dd>{{ event.tag }}</dd>
                    
                    <dt>Câmera:</dt>
                    <dd>{{ event.camera_id }}</dd>
                    
                    <dt>Timestamp de Recebimento:</dt>
                    <dd>{{ event.timestamp }}</dd>
                    
                    <dt>Início da Detecção:</dt>
                    <dd>{{ event.start }}</dd>
                    
                    <dt>Fim da Detecção:</dt>
                    <dd>{{ event.end }}</dd>
                    
                    <dt>Duração:</dt>
                    <dd>{{ event.duration }}</dd>
                    
                    <dt>Coordenadas Iniciais:</dt>
                    <dd>{{ event.coord_initial }}</dd>
                    
                    <dt>Coordenadas Finais:</dt>
                    <dd>{{ event.coord_end }}</dd>
                </dl>
            </div>
            
            <a href="/" class="btn">Voltar para lista</a>
        </div>
    </div>
</body>
</html>
"""

# Escrever os arquivos
with open("static/css/style.css", "w") as css_file:
    css_file.write(css_content)

with open("templates/index.html", "w") as index_file:
    index_file.write(index_html)

with open("templates/event_detail.html", "w") as detail_file:
    detail_file.write(event_detail_html)

# Função principal para iniciar o servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
