import asyncio
import time
import datetime
import cv2
import numpy as np
import traceback
from typing import List, Dict, Any, Set, Tuple, Optional
from ultralytics import YOLO

from app.core.shared_state import active_streams, object_trackers
from app.utils.logging_utils import setup_logger
from app.utils.image_utils import convert_frame_to_bytes
from app.api.models.camera import StreamConfig, CameraInfo
from app.api.models.event import Event
from app.external.event_api import send_event

logger = setup_logger("detection_service")

async def extract_detections(result) -> List[Dict[str, Any]]:
    """
    Extrai as informações (bbox, classe e id) de uma detecção do YOLO.

    Args:
        result: Resultado da detecção YOLO

    Returns:
        Lista de informações de detecção (track_id, class_name, bbox)
    """
    detections = []

    boxes = result.boxes  

    if hasattr(boxes, "id") and boxes.id is not None:
        for i, box in enumerate(boxes):
            try:

                # ID do objeto, classe e coordenadas
                track_id = int(box.id.item())
                cls = int(box.cls.item())
                class_name = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])

                # Testar se a bbox é valida
                if x2 > x1 and y2 > y1:
                    detections.append(
                        {
                            "track_id": track_id,
                            "class_name": class_name,
                            "bbox": (x1, y1, x2, y2),
                        }
                    )
                else:
                    logger.warning(
                        f"bbox inválida detectado: {(x1, y1, x2, y2)} para objeto {track_id}"
                    )

            except Exception as e:
                logger.error(f"Erro ao extrair detecção: {e}")

    return detections


def initialize_tracker_for_camera(camera_id: int) -> None:
    """
    Inicializa o dicionario de objetos para uma câmera específica, se ainda não existir.

    Args:
        camera_id: ID da câmera
    """
    if camera_id not in object_trackers:
        object_trackers[camera_id] = {}


async def update_tracked_object(
    camera_id: int, track_id: int, class_name: str, bbox: tuple, frame: np.ndarray
) -> None:
    """
    Atualiza as informações de detecção de um objeto.

    Args:
        camera_id: ID da câmera
        track_id: ID de rastreamento do objeto
        class_name: Nome da classe do objeto detectado
        bbox: Coordenadas do retângulo delimitador (x1, y1, x2, y2)
        frame: Frame de vídeo atual
    """
    is_new_object = track_id not in object_trackers[camera_id]

    current_time = datetime.datetime.now().isoformat()

    # Testar se a bbox é valida
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        logger.warning(f"bbox inválida para objeto {track_id}: {bbox}")
        return

    if is_new_object:
        try:
            # Se objeto é novo, salva o frame
            frame_bytes = await convert_frame_to_bytes(frame.copy())

            object_trackers[camera_id][track_id] = {
                "class": class_name,
                "last_seen": 0,
                "disappeared": False,
                "bbox": bbox,
                "frame": frame_bytes,
                "first_seen": current_time,
                "last_seen_time": current_time,
                "initial_bbox": bbox,
            }

        except Exception as e:
            logger.error(f"Erro ao armazenar frame para objeto {track_id}: {e}")
            object_trackers[camera_id][track_id] = {
                "class": class_name,
                "last_seen": 0,
                "disappeared": False,
                "bbox": bbox,
                "frame": None,  # Erro armazenando o frame
                "first_seen": current_time,
                "last_seen_time": current_time,
                "initial_bbox": bbox,
            }
    else:
        # Objeto com esse ID já existe. Atualização do objeto.
        current_data = object_trackers[camera_id][track_id]
        object_trackers[camera_id][track_id] = {
            "class": class_name,
            "last_seen": 0,
            "disappeared": False,
            "bbox": bbox,
            "frame": current_data.get("frame"),  # Mantem primeiro frame
            "first_seen": current_data.get("first_seen"),
            "last_seen_time": current_time,
            "initial_bbox": current_data.get("initial_bbox", bbox),
        }


def process_disappearances(
    current_track_ids: set, stream_config: StreamConfig
) -> list:
    """
    Processa objetos rastreados e identifica aqueles que desapareceram.

    Args:
        camera_id: ID da câmera
        current_track_ids: Conjunto de IDs de rastreamento detectados no frame atual
        stream_config: Configuração YOLO com parâmetros de rastreamento

    Returns:
        Lista de objetos desaparecidos
    """

    camera_id = stream_config.camera_id
    disappeared_objects = []

    if camera_id not in object_trackers:
        return disappeared_objects

    for track_id in list(object_trackers[camera_id].keys()):
        # Aumenta o contador dos objetos não vistos nesse frame  
        if track_id not in current_track_ids:
            object_trackers[camera_id][track_id]["last_seen"] += 1

            # Verifica se o objeto deve ser considerado desaparecido
            if (
                object_trackers[camera_id][track_id]["last_seen"]
                >= stream_config.frames_before_disappearance
                and not object_trackers[camera_id][track_id]["disappeared"]
            ):
                object_trackers[camera_id][track_id]["disappeared"] = True

                # Adiciona objeto como desaparecido
                disappeared_objects.append(
                    {
                        "track_id": track_id,
                        "class": object_trackers[camera_id][track_id]["class"],
                        "last_bbox": object_trackers[camera_id][track_id]["bbox"],
                        "last_seen_time": object_trackers[camera_id][track_id].get(
                            "last_seen_time", datetime.datetime.now().isoformat()
                        ),
                    }
                )

            # Remove dos objetos salvos
            if (
                object_trackers[camera_id][track_id]["last_seen"]
                >= stream_config.frames_before_disappearance * 2
            ):
                del object_trackers[camera_id][track_id]

    return disappeared_objects


def log_disappearances(camera_id: int, disappeared_objects: list) -> None:
    """
    Printa informações sobre objetos desaparecidos.

    Args:
        camera_id: ID da câmera
        disappeared_objects: Lista de objetos desaparecidos
    """
    for obj in disappeared_objects:
        tracker_data = object_trackers[camera_id][obj["track_id"]]
        initial_time = tracker_data.get(
            "first_seen", datetime.datetime.now().isoformat()
        )
        first_seen = datetime.datetime.fromisoformat(initial_time)
        duration = (datetime.datetime.now() - first_seen).total_seconds()
        message = f"{obj['class']} ID:{obj['track_id']} duração: {duration:.2f}s"
        logger.info(f"Câmera {camera_id}: {message}")


async def send_disappearance_events(camera_id: int, disappeared_objects: list) -> None:
    """
    Envia eventos para outro endpoint. Um para cada objeto desaparecido.

    Args:
        camera_id: ID da câmera
        disappeared_objects: Lista de objetos desaparecidos
    """
    for obj in disappeared_objects:
        try:
            track_id = obj["track_id"]

            tracker_data = object_trackers[camera_id][track_id]

            frame_bytes = tracker_data.get("frame")

            if frame_bytes is None:
                logger.warning(
                    f"Nenhum frame armazenado para objeto {track_id}, ignorando evento"
                )
                continue

            try:
                bbox = tracker_data.get("initial_bbox")

                if not bbox:
                    logger.warning(
                        f"Nenhuma informação de retângulo delimitador para objeto {track_id}"
                    )
                    continue

                x1, y1, x2, y2 = bbox

                initial_time = tracker_data.get(
                    "first_seen", datetime.datetime.now().isoformat()
                )
                last_seen_time = tracker_data.get(
                    "last_seen_time", datetime.datetime.now().isoformat()
                )

                event = Event(
                    camera_id=camera_id,
                    start=initial_time,
                    end=last_seen_time,
                    event_type="objects",
                    tag=obj["class"],
                    coord_initial=(x1, y1),
                    coord_end=(x2, y2),
                    print=frame_bytes,
                )

                await send_event(event)

            except Exception as img_error:
                logger.error(
                    f"Erro ao processar imagem para objeto {track_id}: {img_error}"
                )

        except Exception as e:
            logger.error(f"Erro ao enviar evento de desaparecimento: {e}")


async def process_frame(
    model, stream_config: StreamConfig, frame: np.ndarray
) -> None:
    """
    Processa um único frame de uma câmera e delega para análises de detecção.
    Gerencia o pipeline de inferência e análise.

    Args:
        model: Instância do modelo YOLO
        camera_id: ID da câmera de origem
        stream_config: Configuração para detecção YOLO
        frame: O frame de imagem a ser processado
    """
    try:
        # Converter nomes de classes para índices (YOLO recebe indices)
        # TODO: evitar fazer isso para todos os frames
        camera_id = stream_config.camera_id
        if stream_config.classes:
            name_to_index = {v: k for k, v in model.names.items()}
            class_ids = [
                name_to_index[name]
                for name in stream_config.classes
                if name in name_to_index
            ]
        else:
            class_ids = None

        # Cria uma thread separada para rodar o YOLO 
        results = await asyncio.to_thread(
            model.track,
            source=frame,
            persist=True,
            conf=stream_config.confidence_threshold,
            iou=stream_config.iou,
            verbose=False,
            tracker=stream_config.tracker_model,
            classes=class_ids,  # Se for None, o YOLO apenas ignora. 
        )

        current_track_ids = set()

        for result in results:
            detections = await extract_detections(result)

            for detection in detections:
                current_track_ids.add(detection["track_id"])

                # Atualiza objeto detectado com o frame atual e informações de detecção
                await update_tracked_object(
                    camera_id,
                    detection["track_id"],
                    detection["class_name"],
                    detection["bbox"],
                    frame,
                )


        # Determina objetos desaparecidos
        disappeared_objects = process_disappearances(
          current_track_ids, stream_config
        )

        if disappeared_objects:
            log_disappearances(camera_id, disappeared_objects)

            await send_disappearance_events(camera_id, disappeared_objects)

    except Exception as e:
        logger.error(f"Erro ao processar frame para câmera {camera_id}: {e}")


async def process_camera_stream(
    camera_info: CameraInfo, stream_config: StreamConfig
) -> None:
    """
    Executa detecção de objetos contínua em um stream de vídeo de câmera.
    Esta função gerencia a conexão e delega tarefas de detecção para funções auxiliares.
    Executa em segundo plano (para cada strean) até que a stream/camera seja desativada ou a conexão falhe.

    Args:
        camera_info: Objeto CameraInfo contendo ID, URL e informações de status da câmera.
        stream_config: Configuração para o modelo YOLO e parâmetros de detecção.
    """
    cam_id = camera_info.camera_id
    last_frame_time = 0
    frame_interval = 1.0 / stream_config.frames_per_second
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    reconnect_delay = 1  # Atraso inicial em segundos

    # Carrega o modelo YOLO
    local_model = YOLO(stream_config.detection_model_path)

    capture = None

    initialize_tracker_for_camera(cam_id)

    try:
        # Continua procrssando enquanto a câmera estiver ativa no dicionário de streams ativos
        while cam_id in active_streams and active_streams[cam_id]["active"]:

            if capture is None or not capture.isOpened():
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(
                        f"Falha ao conectar à câmera {cam_id} após {max_reconnect_attempts} tentativas. Desativando câmera."
                    )
                    active_streams[cam_id]["active"] = False
                    break

                # Backoff exponencial 
                if reconnect_attempts > 0:
                    backoff_delay = reconnect_delay * (2 ** (reconnect_attempts - 1))
                    capped_delay = min(
                        backoff_delay, 30
                    )  # Limita delay a 30 segundos
                    logger.info(
                        f"Tentativa de reconexão {reconnect_attempts}/{max_reconnect_attempts} para câmera {cam_id}. "
                        f"Aguardando {capped_delay:.1f} segundos antes de tentar novamente."
                    )
                    await asyncio.sleep(capped_delay)

                logger.info(f"Conectando à câmera {cam_id} na URL: {camera_info.url}.")
                capture = cv2.VideoCapture(camera_info.url)

                if not capture.isOpened():
                    reconnect_attempts += 1
                    logger.warning(
                        f"Falha ao conectar à câmera {cam_id}. Tentativa {reconnect_attempts}/{max_reconnect_attempts}."
                    )

                    # Libera captura
                    if capture is not None:
                        capture.release()
                        capture = None
                    continue
                else:
                    # Conexão bem-sucedida
                    if reconnect_attempts > 0:
                        logger.info(
                            f"Reconectado com sucesso à câmera {cam_id} após {reconnect_attempts} tentativas."
                        )
                    reconnect_attempts = 0

            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time

            # Se não for hora de processar novo frame
            if time_since_last_frame < frame_interval:
                # Sleep até hora do proximo frame para evitar CPU bounding
                sleep_time = max(0.05, frame_interval - time_since_last_frame)
                await asyncio.sleep(sleep_time)  
                continue

            ret, frame = capture.read()
            if not ret:
                logger.warning(
                    f"Falha ao ler frame da câmera {cam_id}. Tentando reconectar."
                )
                capture.release()
                capture = None
                reconnect_attempts += 1
                continue

            last_frame_time = current_time

            # Processar frame atual
            await process_frame(local_model, stream_config, frame)

    except Exception as exc:
        logger.error(f"Erro ao processar stream {cam_id}: {exc}")
        traceback.print_exc()  
    finally:
        if capture is not None and capture.isOpened():
            capture.release()
        if cam_id in active_streams:
            active_streams[cam_id]["active"] = False
        logger.info(f"Processamento de stream para câmera {cam_id} encerrado.")
