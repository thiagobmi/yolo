from pydantic import BaseModel
from typing import Tuple, List

class Event(BaseModel):
    """Informações de um evento."""
    camera_id: int
    start: str  # tempo de inicio do evento
    end: str    #  tempo de final do evento
    event_type: str  # identificacao do tipo de evento (ex: "objects")
    tag: str  # classe do objeto observado (carro, pessoa, etc)
    coord_initial: Tuple[int, int]  # coordenada inicial do "quadrado" que identifica o objeto
    coord_end: Tuple[int, int]  # coordenada final do "quadrado" que identifica o objeto
    print: bytes  # print da imagem analisada
