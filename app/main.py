import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Dict, Any, List

from app.config import settings
from app.api.routes import cameras
from app.utils.logging_utils import setup_logger
from app.external.nuv_api import get_camera_info,initialize_nuv_api
from app.api.models.camera import CameraInfo, StreamConfig
from app.config.settings import SPECIFICATIONS_PATH

# logger principal
logger = setup_logger("main")

async def lifespan_event(app: FastAPI):
    """
    Gerencia eventos de inicialização da aplicação.

    Args:
        app: A instância da aplicação FastAPI.
    """
    logger.info("Iniciando aplicação")

    initialize_nuv_api(SPECIFICATIONS_PATH)
    
    yield # Passa o controle da aplicação
    
    logger.info("Encerrando aplicação")

# Cria app FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan_event
)

# inclui rota das cameras
app.include_router(cameras.router)

@app.get("/")
def read_root():
    """Rota raiz da API."""
    return {"detail": "NUVYOLO", "version": settings.API_VERSION}
