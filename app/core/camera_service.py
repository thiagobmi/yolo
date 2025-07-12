from typing import Dict, Any, Optional, List
import datetime
import asyncio

from app.core.shared_state import active_streams, object_trackers
from app.utils.logging_utils import setup_logger
from app.api.models.camera import CameraInfo, StreamConfig
from app.external.nuv_api import get_camera_info

logger = setup_logger("camera_service")

async def start_monitoring_camera(
    stream_config: StreamConfig
) -> Dict[str, Any]:
    """
    Inicia o monitoramento e detecção de objetos para uma câmera específica.
    
    Args:
        camera_id: ID da câmera a ser monitorada.
        stream_config: Parâmetros de configuração do stream incluindo dispositivo, modelos,
                     e configurações de detecção.
                     
    Returns:
        Dict contendo detalhes de status e informações da câmera.
    """

    camera_id = stream_config.camera_id
    # Verifica se a câmera já está sendo monitorada
    if camera_id in active_streams and active_streams[camera_id]["active"]:
        return {"detail": f"Câmera {camera_id} já está sendo monitorada", "camera": None}

    camera_info = await get_camera_info(camera_id)
    if camera_info is None:
        return {"detail": f"Câmera {camera_id} não encontrada", "camera": None}

    # Registra a câmera como ativa
    active_streams[camera_id] = {
        "active": True,
        "info": camera_info.model_dump(),
        "options": stream_config.model_dump(),
        "started_at": datetime.datetime.now().isoformat(),
    }
    
    return {"detail": f"Iniciado monitoramento para câmera {camera_id}", "camera": camera_info}

async def stop_monitoring_camera(camera_id: int) -> Dict[str, str]:
    """
    Interrompe o monitoramento de uma câmera específica.
    
    Args:
        camera_id: ID da câmera para interromper o monitoramento.
        
    Returns:
        Mensagem de status.
    """
    if camera_id in active_streams:
        active_streams[camera_id]["active"] = False

        # Esperar tarefa terminar
        await asyncio.sleep(0.5)
        
        # Remover do dicionário de streams ativos
        if camera_id in active_streams:
            del active_streams[camera_id]
            
        if camera_id in object_trackers:
            del object_trackers[camera_id]
            
        return {"detail": f"Monitoramento interrompido para câmera {camera_id}"}
    else:
        return {"detail": f"Câmera {camera_id} não está sendo monitorada"}

async def stop_all_monitoring() -> Dict[str, Any]:
    """
    Interrompe o monitoramento para todas as câmeras ativas.
    
    Returns:
        Dict com informações sobre as câmeras que foram interrompidas
    """
    stopped_cameras = []
    camera_ids = list(active_streams.keys())

    for camera_id in camera_ids:
        camera_info = {
            "camera_id": camera_id,
            "started_at": active_streams[camera_id]["started_at"],
        }
        
        result = await stop_monitoring_camera(camera_id)
        if "detail" in result and "interrompido" in result["detail"]:
            stopped_cameras.append(camera_info)

    total_stopped = len(stopped_cameras)

    if total_stopped > 0:
        return {
            "detail": f"Monitoramento interrompido com sucesso para {total_stopped} câmera(s)",
            "cameras": stopped_cameras
        }
    else:
        return {"detail": "Nenhuma câmera ativa para interromper", "cameras": []}

def get_monitored_cameras() -> List[Dict[str, Any]]:
    """
    Retorna informações sobre todas as câmeras atualmente monitoradas.
    
    Returns:
        Lista de detalhes de câmeras ativas.
    """
    result = []
    for camera_id, camera_data in active_streams.items():
        camera_info = {
            "camera_id": camera_id,
            "active": camera_data["active"],
            "started_at": camera_data["started_at"],
            "url": camera_data["info"].get("url", ""),
        }
        result.append(camera_info)
        
    return result
