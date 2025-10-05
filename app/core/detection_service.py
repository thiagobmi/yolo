import threading
import time
import datetime
import cv2
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from ultralytics import YOLO
from collections import Counter
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from app.core.shared_state import active_streams, object_trackers
from app.utils.logging_utils import setup_logger
from app.utils.image_utils import convert_frame_to_bytes, draw_bounding_box
from app.api.models.camera import StreamConfig, CameraInfo
from app.api.models.event import Event
from app.external.event_api import send_event
from app.config import settings

logger = setup_logger("detection_service")

# Cache para converter classes para IDs
_class_mapping_cache = {}
STREAM_FPS = 30

# Thread pool para envio de eventos
event_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="event_sender")


def initialize_tracker_for_camera(camera_id: int) -> None:
    """
    Inicializa o dicionario de objetos para uma câmera específica, se ainda não existir.
    """
    if camera_id not in object_trackers:
        object_trackers[camera_id] = {}


def extract_detections(result, scale_factor=1.0) -> List[Dict[str, Any]]:
    """
    Extrai as informações (bbox, classe, id e confiança) de uma detecção do YOLO.
    Ajusta as bounding boxes caso o frame tenha sido redimensionado.
    """
    detections = []

    boxes = result.boxes

    if hasattr(boxes, "id") and boxes.id is not None:
        for i, box in enumerate(boxes):
            try:
                track_id = int(box.id.item())
                cls = int(box.cls.item())
                class_name = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                if scale_factor != 1.0:
                    x1, y1, x2, y2 = (
                        int(x1 / scale_factor),
                        int(y1 / scale_factor),
                        int(x2 / scale_factor),
                        int(y2 / scale_factor),
                    )
                conf = float(box.conf.item())

                if x2 > x1 and y2 > y1:
                    detections.append(
                        {
                            "track_id": track_id,
                            "class_name": class_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                        }
                    )
                else:
                    logger.warning(
                        f"bbox inválida detectado: {(x1, y1, x2, y2)} para objeto {track_id}"
                    )

            except Exception as e:
                logger.error(f"Erro ao extrair detecção: {e}")

    return detections


def update_tracked_object(
    camera_id: int,
    track_id: int,
    class_name: str,
    bbox: tuple,
    frame: np.ndarray,
    confidence: float,
) -> None:
    """
    Atualiza as informações de detecção de um objeto com redimensionamento configurável.
    """
    is_new_object = track_id not in object_trackers[camera_id]
    current_time = datetime.datetime.now().isoformat()

    x1, y1, x2, y2 = bbox

    # Testar se a bbox é valida
    if x1 >= x2 or y1 >= y2:
        return

    if is_new_object:
        try:
            height, width = frame.shape[:2]

            # Redimensiona o frame para economizar memória. width = WIDTH_RESIZE
            if width > settings.WIDTH_RESIZE:
                scale = settings.WIDTH_RESIZE / width
                new_width = settings.WIDTH_RESIZE
                new_height = int(height * scale)
                frame_resized = cv2.resize(
                    frame,
                    (new_width, new_height),
                    # interpolation=cv2.INTER_NEAREST
                )

                bbox_for_frame = (
                    int(x1 * scale),
                    int(y1 * scale),
                    int(x2 * scale),
                    int(y2 * scale),
                )
            else:
                frame_resized = frame
                bbox_for_frame = bbox

            # Converter o frame. qualidade do jpeg = QUALITY_CONVERT.
            frame_bytes = convert_frame_to_bytes(
                frame_resized, settings.QUALITY_CONVERT
            )

            object_trackers[camera_id][track_id] = {
                "class": class_name,
                "last_seen": 0,
                "disappeared": False,
                "bbox": bbox,
                "frame": frame_bytes,
                "bbox_for_frame": bbox_for_frame,
                "detection_history": [
                    {"class": class_name, "confidence": confidence}
                ],  # Salva também a classe e conf de cada detecção.
                "first_seen": current_time,
                "last_seen_time": current_time,
                "initial_bbox": bbox,
            }

        except Exception as e:
            # Fallback sem frame
            logger.error(f"Erro ao processar frame para objeto {track_id}: {e}")

            object_trackers[camera_id][track_id] = {
                "class": class_name,
                "last_seen": 0,
                "disappeared": False,
                "bbox": bbox,
                "frame": None,
                "bbox_for_frame": None,
                "first_seen": current_time,
                "detection_history": [{"class": class_name, "confidence": confidence}],
                "last_seen_time": current_time,
                "initial_bbox": bbox,
            }
    else:
        # Objeto com esse ID já existe. Atualização do objeto.
        object_trackers[camera_id][track_id].update(
            {
                "last_seen": 0,
                "bbox": bbox,
                "last_seen_time": current_time,
            }
        )
        object_trackers[camera_id][track_id]["detection_history"].append(
            {"class": class_name, "confidence": confidence}
        )



def validate_detection_consistency(
    detection_history: list, min_percentage: float = 0.7
) -> tuple[bool, str]:
    """
    Verifica se pelo menos min_percentage das detecções são da mesma classe.
    """
    if not detection_history:
        return False, ""

    # Conta as ocorrências de cada classe
    class_counts = Counter(detection["class"] for detection in detection_history)

    # encontra a classe mais comum
    most_common_class, most_common_count = class_counts.most_common(1)[0]

    # calcula a porcentagem da classe mais comum
    total_detections = len(detection_history)
    percentage = most_common_count / total_detections

    # Verifica se tem no mínimo essa porcentagem da mesma classe
    is_valid = percentage >= min_percentage

    return is_valid, most_common_class


def send_single_event(
    obj_data: dict, stream_config: StreamConfig, camera_id: int
) -> bool:
    """
    Envia um único evento para o endpoint 
    """
    try:
        track_id = obj_data["track_id"]
        frame_bytes = obj_data.get("frame")
        bbox_for_frame = obj_data.get("bbox_for_frame")

        if not frame_bytes or not bbox_for_frame:
            logger.warning(f"Dados faltando para objeto {track_id}")
            return False

        initial_time = obj_data.get("first_seen", datetime.datetime.now().isoformat())
        last_seen_time = obj_data.get(
            "last_seen_time", datetime.datetime.now().isoformat()
        )

        x1, y1, x2, y2 = bbox_for_frame

        detection_history = obj_data["detection_history"]

        # mínimo de detecções >= min_track
        min_len = len(detection_history) >= stream_config.min_track_frames

        # 70% ou + das detecções são da mesma classe
        is_class_consistent, main_class = validate_detection_consistency(
            detection_history, min_percentage=0.7  # stream_config.min_class_percentage
        )

        if not min_len or not is_class_consistent:
            return False

        event = Event(
            camera_id=camera_id,
            start=initial_time,
            end=last_seen_time,
            event_type="objects",
            tag=main_class,
            coord_initial=(x1, y1),
            coord_end=(x2, y2),
            print=frame_bytes,
        )

        # Chama a função síncrona de envio
        return send_event(event)

    except Exception as e:
        logger.error(f"Erro ao processar evento {track_id}: {e}")
        return False


def send_disappearance_events(
    stream_config: StreamConfig, camera_id: int, disappeared_objects: list
) -> None:
    """
    Envia eventos para o endpoint usando threading. 
    """
    if not disappeared_objects:
        return

    # Envia eventos em paralelo 
    futures = []
    for obj in disappeared_objects:
        future = event_executor.submit(send_single_event, obj, stream_config, camera_id)
        futures.append(future)

    # for future in as_completed(futures, timeout=5.0):
    #     try:
    #         result = future.result()
    #         if not result:
    #             logger.warning("Falha ao enviar evento")
    #     except Exception as e:
    #         logger.error(f"Erro no envio de evento: {e}")


def process_disappearances(current_track_ids: set, stream_config: StreamConfig) -> list:
    """
    Processa objetos rastreados e identifica aqueles que desapareceram.
    """
    camera_id = stream_config.camera_id
    disappeared_objects = []

    if camera_id not in object_trackers:
        return disappeared_objects

    objects_to_delete = []

    for track_id in list(object_trackers[camera_id].keys()):
        if track_id not in current_track_ids:
            object_trackers[camera_id][track_id]["last_seen"] += 1

            if (
                object_trackers[camera_id][track_id]["last_seen"]
                >= stream_config.frames_before_disappearance
                and not object_trackers[camera_id][track_id]["disappeared"]
            ):
                object_trackers[camera_id][track_id]["disappeared"] = True

                tracker_data = object_trackers[camera_id][track_id]

                disappeared_objects.append(
                    {
                        "track_id": track_id,
                        "class": tracker_data["class"],
                        "last_bbox": tracker_data["bbox"],
                        "last_seen_time": tracker_data.get(
                            "last_seen_time", datetime.datetime.now().isoformat()
                        ),
                        "first_seen": tracker_data.get("first_seen"),
                        "frame": tracker_data.get("frame"),
                        "initial_bbox": tracker_data.get("initial_bbox"),
                        "bbox_for_frame": tracker_data.get("bbox_for_frame"),
                        "detection_history": tracker_data.get("detection_history"),
                    }
                )

            # Deleta objetos que já estão sem aparecer há muitos frames.
            if (
                object_trackers[camera_id][track_id]["last_seen"]
                >= stream_config.frames_before_disappearance
            ):
                objects_to_delete.append(track_id)

    for track_id in objects_to_delete:
        if track_id in object_trackers[camera_id]:
            del object_trackers[camera_id][track_id]

    return disappeared_objects


def process_frame(model, stream_config: StreamConfig, frame: np.ndarray) -> None:
    """
    Processa um único frame de uma câmera e delega para análises de detecção.
    """
    scale_factor = 1.0
    try:
        camera_id = stream_config.camera_id

        # Converter nomes de classes para índices. Cache para evitar processamento desnecessário.
        cache_key = f"{camera_id}_{hash(tuple(stream_config.classes or []))}"
        if cache_key not in _class_mapping_cache:
            if stream_config.classes:
                name_to_index = {v: k for k, v in model.names.items()}
                class_ids = [
                    name_to_index[name]
                    for name in stream_config.classes
                    if name in name_to_index
                ]
            else:
                class_ids = None
            _class_mapping_cache[cache_key] = class_ids
        else:
            class_ids = _class_mapping_cache[cache_key]

        try:
            height, width = frame.shape[:2]
            target_size = 1280  # or 1024

            if settings.RESIZE_FRAME and width < target_size:
                scale_factor = target_size / width
                scaled_height = int(height * scale_factor)
                scaled_frame = cv2.resize(frame, (target_size, scaled_height))
            else:
                scaled_frame = frame
                scale_factor = 1.0

            results = model.track(
                source=scaled_frame,
                persist=True,
                conf=stream_config.confidence_threshold,
                iou=stream_config.iou,
                verbose=False,
                tracker=stream_config.tracker_model,
                classes=class_ids,
            )

        except Exception as e:
            logger.error(f"Erro na inferência YOLO para câmera {camera_id}: {e}")
            return

        current_track_ids = set()

        for result in results:
            detections = extract_detections(result, scale_factor)

            for detection in detections:
                current_track_ids.add(detection["track_id"])

                update_tracked_object(
                    camera_id,
                    detection["track_id"],
                    detection["class_name"],
                    detection["bbox"],
                    frame,
                    detection["confidence"],
                )

        disappeared_objects = process_disappearances(current_track_ids, stream_config)

        if disappeared_objects:
            # Envia eventos usando thread separada
            threading.Thread(
                target=send_disappearance_events,
                args=(stream_config, camera_id, disappeared_objects),
                daemon=True,
            ).start()

    except Exception as e:
        logger.error(f"Erro ao processar frame para câmera {camera_id}: {e}")


def process_camera_stream(camera_info: CameraInfo, stream_config: StreamConfig) -> None:
    """
    Loop principal de processamento de stream da câmera com lógica de reconexão.
    """
    cam_id = camera_info.camera_id
    local_model = YOLO(stream_config.detection_model_path)

    initialize_tracker_for_camera(cam_id)

    capture = None
    frames_processed = 0
    start_time = time.time()
    reconnect_attempts = 0
    max_reconnect_attempts = settings.MAX_RECONNECT_ATTEMPTS or 5
    reconnect_delay = settings.INITIAL_RECONNECT_DELAY or 1

    try:
        while cam_id in active_streams and active_streams[cam_id]["active"]:

            # Lógica de conexão/reconexão
            if capture is None or not capture.isOpened():
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(
                        f"Falha ao conectar à câmera {cam_id} após {max_reconnect_attempts} tentativas"
                    )
                    active_streams[cam_id]["active"] = False
                    break

                if reconnect_attempts > 0:
                    backoff_delay = min(
                        reconnect_delay * (2 ** (reconnect_attempts - 1)), 30
                    )
                    logger.info(
                        f"Aguardando {backoff_delay:.1f}s antes de reconectar câmera {cam_id}"
                    )
                    time.sleep(backoff_delay)

                try:
                    capture = cv2.VideoCapture(camera_info.url)

                    if capture.isOpened():
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        logger.info(f"Câmera {cam_id} conectada com sucesso")
                        reconnect_attempts = 0
                    else:
                        logger.warning(f"Falha ao abrir câmera {cam_id}")
                        reconnect_attempts += 1
                        if capture:
                            capture.release()
                            capture = None
                        continue

                except Exception as e:
                    logger.error(f"Erro ao conectar câmera {cam_id}: {e}")
                    reconnect_attempts += 1
                    if capture:
                        capture.release()
                        capture = None
                    continue

            # Leitura do frame
            try:
                ret, frame = capture.read()
                if not ret or frame is None:
                    logger.warning(f"Falha ao ler frame da câmera {cam_id}")
                    capture.release()
                    capture = None
                    reconnect_attempts += 1
                    continue

            except Exception as read_error:
                logger.error(f"Erro ao ler frame da câmera {cam_id}: {read_error}")
                capture.release()
                capture = None
                reconnect_attempts += 1
                continue

            frames_processed += 1

            # Processar a cada N frames 
            if frames_processed % (STREAM_FPS // stream_config.frames_per_second) == 0:
                try:
                    process_frame(local_model, stream_config, frame)
                except Exception as e:
                    logger.error(f"Erro ao processar frame {frames_processed}: {e}")

            # Log de status 
            # if frames_processed % 300 == 0:  # A cada 10 segundos aproximadamente
            #     elapsed = time.time() - start_time
            #     fps = frames_processed / elapsed
            #     logger.info(
            #         f"CÂMERA {cam_id}: {frames_processed} frames, {fps:.1f} fps média"
            #     )

            time.sleep(0.001)  # 1ms 

    finally:
        if capture is not None and capture.isOpened():
            capture.release()
        if cam_id in active_streams:
            active_streams[cam_id]["active"] = False

        elapsed = time.time() - start_time
        final_fps = frames_processed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Câmera {cam_id} encerrada - {frames_processed} frames em {elapsed:.1f}s ({final_fps:.1f} fps)"
        )


def start_camera_processing(
    camera_info: CameraInfo, stream_config: StreamConfig
) -> threading.Thread:
    """
    Inicia o processamento de uma câmera em uma thread separada.
    """
    thread = threading.Thread(
        target=process_camera_stream,
        args=(camera_info, stream_config),
        name=f"camera_{camera_info.camera_id}",
        daemon=False,
    )
    thread.start()
    return thread


def stop_all_cameras():
    """
    Para o processamento de todas as câmeras.
    """
    for cam_id in active_streams:
        active_streams[cam_id]["active"] = False

    time.sleep(2)

    # Finaliza o thread pool de eventos
    event_executor.shutdown(wait=True)
