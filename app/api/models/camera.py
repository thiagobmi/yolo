from pydantic import BaseModel
from typing import Optional, List

class CameraInfo(BaseModel):
    """Informações da câmera."""
    camera_id: int
    url: str
    active: bool

class StreamConfig(BaseModel):
    """Configurações YOLO para uma stream."""
    camera_id: int
    device: str
    detection_model_path: str
    classes: Optional[List[str]] = None
    tracker_model: str
    frames_per_second: int
    frames_before_disappearance: int
    confidence_threshold: float
    iou: float

class CameraResponse(BaseModel):
    """Resposta das operações de câmera."""
    detail: str
    camera: Optional[CameraInfo] = None

class MonitoredCamerasResponse(BaseModel):
    """Resposta com lista de câmeras monitoradas."""
    cameras: List[dict]
