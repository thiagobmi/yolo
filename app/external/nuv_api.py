import requests
import json
from typing import Dict, Any, Optional
from app.utils.logging_utils import setup_logger
from app.config import settings
from app.api.models.camera import CameraInfo
from app.external.nuv_api_wrapper import NuvAPIWrapper


logger = setup_logger("nuv_api")

def populate_nuv_api_wrapper(specifications: dict) -> None:
    """Populates the Nuv API Wrapper according to the passed specifications.

    Args:
        specifications (dict): Benchmark specifications.
    """
    NuvAPIWrapper.origin_ip = specifications["api_specifications"]["origin_ip"]
    NuvAPIWrapper.origin_username = specifications["api_specifications"]["origin_username"]
    NuvAPIWrapper.origin_password = specifications["api_specifications"]["origin_password"]
    NuvAPIWrapper.edge_ip = specifications["api_specifications"]["edge_ip"]
    NuvAPIWrapper.org_username = specifications["api_specifications"]["org_username"]
    NuvAPIWrapper.org_password = specifications["api_specifications"]["org_password"]
    NuvAPIWrapper.org_name = specifications["api_specifications"]["org_name"]
    NuvAPIWrapper.org_domain = specifications["api_specifications"]["org_domain"]
    NuvAPIWrapper.requests_response_times = []

    # Printing the nuv API Wrapper attributes defined according to the passed specifications
    logger.info(f"Origin IP: {NuvAPIWrapper.origin_ip}")
    logger.info(f"Edge IP: {NuvAPIWrapper.edge_ip}")
    logger.info(f"Org Name: {NuvAPIWrapper.org_name}")
    logger.info(f"Org Domain: {NuvAPIWrapper.org_domain}")

    # NuvAPIWrapper.run_request(method_name="get_manager_token")
    # NuvAPIWrapper.run_request(method_name="get_org_id")
    # NuvAPIWrapper.run_request(method_name="get_tribe_id")


def read_specifications(json_file_path: str):
    """
    Parses the program specifications from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file.

    Returns:
        specifications (dict): Dictionary containing the JSON data.
    """
    with open(json_file_path, "r") as file:
        specifications = json.load(file)
    return specifications



def initialize_nuv_api(specifications_path: str):
    # Setting basic environment variables

    specifications = read_specifications(specifications_path)
    populate_nuv_api_wrapper(specifications)
    
    # Initialize API wrapper
    # TODO: Getting camera info without NuvAPIWrapper?
    return NuvAPIWrapper.run_request(method_name="get_manager_token")


def get_camera_info_api(camera_id:int):
    return NuvAPIWrapper.run_request(method_name="get_camera", method_parameters={"camera_id": camera_id})

# def get_camera_info_api(cam_id: int) -> Dict[str, Any]:
#     """
#     Obtém informações da câmera através da API do NUV.
#
#     Args:
#         cam_id: ID da câmera a ser consultada.
#
#     Returns:
#         Dicionário com as informações da câmera ou mensagem de erro.
#     """
#     try:
#         logger.info(f"Consultando informações da câmera {cam_id} via API NUV")
#
#         # Retorno mockado 
#         return {
#             "id": cam_id,
#             "stream_url": f"rtmp://{{domain}}/{cam_id}",
#             "is_active": True
#         }
#
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Erro ao obter informações da câmera {cam_id}: {e}")
#         return {"detail": f"Erro ao consultar a câmera: {str(e)}"}
#     except Exception as e:
#         logger.error(f"Erro inesperado ao consultar a câmera {cam_id}: {e}")
#         return {"detail": f"Erro inesperado: {str(e)}"}

async def get_camera_info(cam_id: int) -> Optional[CameraInfo]:
    """
    Usa a API do NUV para recuperar informações de uma câmera.
    
    Args:
        cam_id: ID da câmera para recuperar informações.
        
    Returns:
        Objeto CameraInfo com detalhes da câmera, se encontrada, None caso contrário.
    """
    # if settings.DEBUG:
    # return CameraInfo(
    #     camera_id=cam_id, 
    #     url=f"rtmp://localhost/stream/{cam_id}", 
    #     active=True
    # )
    
    response = get_camera_info_api(cam_id)
    
    if "detail" in response:
        return None
    
    stream_url = response["stream_url"].replace("{domain}", settings.DOMAIN)
    logger.info(f"URL: {stream_url}")
    
    return CameraInfo(
        camera_id=response["id"],
        url=stream_url,
        active=response["is_active"],
    )



