import requests
from typing import Dict, Any, Optional
import logging
from app.utils.logging_utils import setup_logger
from app.api.models.event import Event
from app.config import settings

logger = setup_logger("event_api")


async def send_event(event: Event) -> Optional[Dict[str, Any]]:
    """
    Envia o evento para o endpoint de eventos.

    Args:
        event: O evento a ser enviado

    Returns:
        Resposta da API, se bem-sucedida, ou None em caso de erro
    """
    try:
        event_viewer_url = settings.SEND_EVENT_URL

        if isinstance(event.print, bytes):
            # Converter para hex 
            print_hex = event.print.hex()
        else:
            # Se já for string, assumir que já está codificado
            print_hex = event.print

        event_dict = {
            "camera_id": event.camera_id,
            "start": event.start,
            "end": event.end,
            "event_type": event.event_type,
            "tag": event.tag,
            "coord_initial": event.coord_initial,
            "coord_end": event.coord_end,
            "print": print_hex,
        }

        response = requests.post(
            event_viewer_url, json=event_dict, timeout=settings.SEND_EVENT_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Evento enviado ID: {result.get('event_id')}")
            return result
        else:
            logger.error(
                f"Falha ao enviar evento para o visualizador. Código de status: {response.status_code}"
            )
            logger.error(f"Resposta: {response.text}")
            return None

    except requests.exceptions.Timeout:
        logger.error("Timeout ao enviar evento para o visualizador de eventos")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de requisição ao enviar evento para o visualizador: {e}")
    except Exception as e:
        logger.error(f"Erro geral ao enviar evento para o visualizador: {e}")

    return None
