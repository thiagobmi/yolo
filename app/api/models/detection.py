from pydantic import BaseModel
from typing import Tuple, List, Dict, Any


class TrackedObject(BaseModel):
    """Informações de um objeto rastreado."""
    class_name: str
    last_seen: int
    disappeared: bool
    bbox: Tuple[int, int, int, int]
    initial_bbox: Tuple[int, int, int, int]
    frame: bytes
    first_seen: str
    last_seen_time: str

# class Detection(BaseModel):
#     """Modelo para detecções de objetos."""
#     track_id: int
#     class_name: str
#     bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

