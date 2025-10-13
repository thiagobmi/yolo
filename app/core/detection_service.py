from os import wait
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

# thread pool para envio de eventos
event_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="event_sender")

# thread pool para conversao de frames
frame_converter_executor = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="frame_converter"
)


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
                        f"bbox inválida detectada: {(x1, y1, x2, y2)} para objeto {track_id}"
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


    is_new_object = track_id not in object_trackers[camera_id]
    current_time = datetime.datetime.now().isoformat()
    x1, y1, x2, y2 = bbox

    if x1 >= x2 or y1 >= y2:
        return

    # calcula área da bbox atual
    current_area = (x2 - x1) * (y2 - y1)

    if is_new_object:

        def process_frame_async():
            try:
                height, width = frame.shape[:2]
                if width > settings.WIDTH_RESIZE:
                    scale = settings.WIDTH_RESIZE / width
                    new_width = settings.WIDTH_RESIZE
                    new_height = int(height * scale)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                    bbox_for_frame = (
                        int(x1 * scale),
                        int(y1 * scale),
                        int(x2 * scale),
                        int(y2 * scale),
                    )
                else:
                    frame_resized = frame
                    bbox_for_frame = bbox

                frame_bytes = convert_frame_to_bytes(
                    frame_resized, settings.QUALITY_CONVERT
                )

                object_trackers[camera_id][track_id].update(
                    {
                        "frame": frame_bytes,
                        "bbox_for_frame": bbox_for_frame,
                    }
                )

            except Exception as e:
                logger.error(f"Erro ao processar frame para objeto {track_id}: {e}")

        object_trackers[camera_id][track_id] = {
            "class": class_name,
            "last_seen": 0,
            "disappeared": False,
            "bbox": bbox,
            "frame": None,
            "bbox_for_frame": None,
            "detection_history": [{"class": class_name, "confidence": confidence}],
            "first_seen": current_time,
            "last_seen_time": current_time,
            "max_bbox_area": current_area,
        }

        frame_converter_executor.submit(process_frame_async)
    else:

        max_area = object_trackers[camera_id][track_id].get("max_bbox_area", 0)

        # só atualiza a imagem se a área for pelo menos 20% maior
        if current_area > max_area * 1.2:
            logger.info("Sybau")

            def process_frame_async():
                try:
                    height, width = frame.shape[:2]
                    if width > settings.WIDTH_RESIZE:
                        scale = settings.WIDTH_RESIZE / width
                        new_width = settings.WIDTH_RESIZE
                        new_height = int(height * scale)
                        frame_resized = cv2.resize(frame, (new_width, new_height))
                        bbox_for_frame = (
                            int(x1 * scale),
                            int(y1 * scale),
                            int(x2 * scale),
                            int(y2 * scale),
                        )
                    else:
                        frame_resized = frame
                        bbox_for_frame = bbox

                    frame_bytes = convert_frame_to_bytes(
                        frame_resized, settings.QUALITY_CONVERT
                    )

                    object_trackers[camera_id][track_id].update(
                        {
                            "frame": frame_bytes,
                            "bbox_for_frame": bbox_for_frame,
                            "max_bbox_area": current_area,
                        }
                    )

                except Exception as e:
                    logger.error(f"Erro ao processar frame para objeto {track_id}: {e}")

            frame_converter_executor.submit(process_frame_async)

        # atualização normal dos outros parametros
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

    # conta as ocorrências de cada classe
    class_counts = Counter(detection["class"] for detection in detection_history)

    # encontra a classe mais comum
    most_common_class, most_common_count = class_counts.most_common(1)[0]

    # calcula a porcentagem da classe mais comum
    total_detections = len(detection_history)
    percentage = most_common_count / total_detections

    # verifica se tem no mínimo essa porcentagem da mesma classe
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

    # envia eventos em paralelo
    # futures = []
    for obj in disappeared_objects:
        event_executor.submit(send_single_event, obj, stream_config, camera_id)
        # future = event_executor.submit(send_single_event, obj, stream_config, camera_id)
        # futures.append(future)

    # for future in as_completed(futures, timeout=5.0):
    #     try:
    #         result = future.result()
    #         if not result:
    #             logger.warning("Falha ao enviar evento")
    #     except Exception as e:
    #         logger.error(f"Erro no envio de evento: {e}")


def process_disappearances(current_track_ids: set, stream_config: StreamConfig) -> list:
    """
    Processa objetos rastreados e identifica os que desapareceram.
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
    Processa um único frame de uma câmera e delega para análises das detecções.
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
    Gerencia a lógica de conexão, reconexão e processamento dos frames
    """

    cam_id = camera_info.camera_id
    local_model = YOLO(stream_config.detection_model_path)
    initialize_tracker_for_camera(cam_id)
    start_time = time.time()

    frame_queue = queue.Queue(maxsize=2)
    should_stop = threading.Event()

    max_reconnect_attempts = settings.MAX_RECONNECT_ATTEMPTS or 5
    reconnect_delay = settings.INITIAL_RECONNECT_DELAY or 2

    def capture_frames():
        """Thread para captura dos frames, com reconexão"""
        capture = None
        reconnect_count = 0

        while not should_stop.is_set():

            # Conexão
            if capture is None or not capture.isOpened():
                if reconnect_count >= max_reconnect_attempts:
                    logger.error(f"Câmera {cam_id}: máximo de reconexões atingido")
                    active_streams[cam_id]["active"] = False
                    break

                if reconnect_count > 0:
                    wait_time = reconnect_delay * reconnect_count
                    logger.info(
                        f"Câmera {cam_id}: aguardando {wait_time}s para reconectar..."
                    )
                    time.sleep(wait_time)

                logger.info(f"Câmera {cam_id}: conectando...")

                try:
                    if capture is not None:
                        capture.release()

                    capture = cv2.VideoCapture(camera_info.url)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if capture.isOpened():
                        connection_time = time.time()
                        logger.info(
                            f"Câmera {cam_id}: conectada com sucesso [{connection_time - start_time}s]"
                        )
                        reconnect_count = 0
                    else:
                        logger.warning(f"Câmera {cam_id}: falha ao conectar")
                        reconnect_count += 1
                        continue

                except Exception as e:
                    logger.error(f"Câmera {cam_id}: erro ao conectar - {e}")
                    reconnect_count += 1
                    continue

            # Lê frame
            try:
                ret, frame = capture.read()

                if not ret or frame is None:
                    logger.warning(f"Câmera {cam_id}: falha ao ler frame")
                    capture.release()
                    capture = None
                    reconnect_count += 1
                    continue

                # Adiciona na queue (descarta frame novo se queue estiver cheia)
                try:
                    frame_queue.put(frame, block=False)
                except queue.Full:
                    pass

            except Exception as e:
                logger.error(f"Câmera {cam_id}: erro na leitura - {e}")
                if capture is not None:
                    capture.release()
                    capture = None
                reconnect_count += 1

        # Cleanup
        if capture is not None:
            capture.release()
        logger.info(f"Câmera {cam_id}: thread de captura encerrada")

    # Inicia thread de captura
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    time_connected = time.time()

    frames_processed = 0

    # Thread principal
    try:
        while cam_id in active_streams and active_streams[cam_id]["active"]:
            try:
                frame = frame_queue.get()
                frames_processed += 1

                if (
                    frames_processed % (STREAM_FPS // stream_config.frames_per_second)
                    == 0
                ):
                    process_frame(local_model, stream_config, frame)

                if frames_processed % 300 == 0:
                    elapsed = time.time() - time_connected
                    fps = frames_processed / elapsed
                    logger.info(
                        f"Câmera {cam_id}: {frames_processed} frames, {fps:.1f} fps"
                    )

            except queue.Empty:
                continue

    finally:
        should_stop.set()
        capture_thread.join(timeout=5)

        elapsed = time.time() - time_connected
        logger.info(
            f"Câmera {cam_id}: encerrada - {frames_processed} frames em {elapsed:.1f}s"
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
    daemon=True  # encerra junto com o processo principal
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

    frame_converter_executor.shutdown(wait=True)
