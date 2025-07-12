from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any, List

from app.api.models.camera import CameraInfo, StreamConfig, CameraResponse, MonitoredCamerasResponse
from app.core.camera_service import (
    start_monitoring_camera, 
    stop_monitoring_camera, 
    stop_all_monitoring, 
    get_monitored_cameras
)
from app.core.detection_service import process_camera_stream
from app.utils.logging_utils import setup_logger

logger = setup_logger("camera_routes")

router = APIRouter(tags=["cameras"])

@router.post("/monitor", response_model=CameraResponse)
async def start_monitoring(
    stream_config: StreamConfig, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Inicia o monitoramento e detecção de objetos para uma câmera específica.
    
    Args:
        camera_id: ID da câmera a ser monitorada.
        stream_config: Parâmetros de configuração do stream incluindo dispositivo, modelos,
                     e configurações de detecção.
        background_tasks: Manipulador de tarefas em segundo plano do FastAPI para processamento assíncrono.
        
    Returns:
        Dict contendo detalhes de status e informações da câmera.
    """
    camera_id = stream_config.camera_id
    try:
        response = await start_monitoring_camera(stream_config)

        # if(stream_config.frames_per_second <= 0):...
        # if(stream_config.frames_before_disappearance <= 0):...
        # if(stream_config.confidence_threshold <= 0):...
        # if(stream_config.iou <= 0):...
        
        if "camera" in response and response["camera"] is not None:
            # Cria uma tarefa em background para cada stream
            background_tasks.add_task(process_camera_stream, response["camera"], stream_config)
            
        return response
    except Exception as exc:
        logger.error(f"Erro ao iniciar monitoramento para câmera {camera_id}: {exc}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@router.post("/stop/{camera_id}")
async def stop_monitoring_route(camera_id: int) -> Dict[str, str]:
    """
    Interrompe o monitoramento de uma câmera específica.
    
    Args:
        camera_id: ID da câmera para interromper o monitoramento.
        
    Returns:
        Dict com mensagem de status.
    """
    try:
        result = await stop_monitoring_camera(camera_id)
        
        if "não está sendo monitorada" in result["detail"]:
            raise HTTPException(
                status_code=404, detail=f"Câmera {camera_id} não está sendo monitorada"
            )
            
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        logger.error(f"Erro ao interromper monitoramento para câmera {camera_id}: {exc}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@router.post("/stop/all")
async def stop_all_monitoring_route() -> Dict[str, Any]:
    """
    Interrompe o monitoramento para todas as câmeras ativas.
    
    Returns:
        Dict com informações sobre as câmeras que foram interrompidas
    """
    try:
        return await stop_all_monitoring()
    except Exception as exc:
        logger.error(f"Erro ao interromper monitoramento de todas as câmeras: {exc}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@router.get("/monitored", response_model=MonitoredCamerasResponse)
async def get_monitored_cameras_route() -> Dict[str, List[Dict[str, Any]]]:
    """
    Retorna informações sobre todas as câmeras atualmente monitoradas.
    
    Returns:
        Dict contendo uma lista de detalhes de câmeras ativas.
    """
    try:
        cameras = get_monitored_cameras()
        return {"cameras": cameras}
    except Exception as exc:
        logger.error(f"Erro ao obter câmeras monitoradas: {exc}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")
