import asyncio
import time
import datetime
import cv2
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from ultralytics import YOLO

from app.core.shared_state import active_streams, object_trackers
from app.utils.logging_utils import setup_logger
from app.utils.image_utils import convert_frame_to_bytes, draw_bounding_box
from app.api.models.camera import StreamConfig, CameraInfo
from app.api.models.event import Event
from app.external.event_api import send_event
from app.config import settings

logger = setup_logger("detection_service")

# Cache para mapeamento de classes
_class_mapping_cache = {}

async def extract_detections(result) -> List[Dict[str, Any]]:
    """
    Extrai as informações (bbox, classe e id) de uma detecção do YOLO.
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
    """
    if camera_id not in object_trackers:
        object_trackers[camera_id] = {}

async def update_tracked_object(
    camera_id: int, track_id: int, class_name: str, bbox: tuple, frame: np.ndarray
) -> None:
    """
    Atualiza as informações de detecção de um objeto com redimensionamento configurável.
    """
    is_new_object = track_id not in object_trackers[camera_id]
    current_time = datetime.datetime.now().isoformat()

    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return

    if is_new_object:
        try:
            height, width = frame.shape[:2]
            
            # Redimensionar usando WIDTH_CONVERT configurável

            if width > settings.WIDTH_CONVERT:
                scale = settings.WIDTH_CONVERT / width
                new_width = settings.WIDTH_CONVERT
                new_height = int(height * scale)
                frame_resized = cv2.resize(
                    frame, (new_width, new_height), 
                    # interpolation=cv2.INTER_NEAREST
                )
                
                bbox_for_frame = (
                    int(x1 * scale), int(y1 * scale), 
                    int(x2 * scale), int(y2 * scale)
                )
            else:
                frame_resized = frame
                bbox_for_frame = bbox
            
            # Usar QUALITY_CONVERT configurável
            frame_bytes = await convert_frame_to_bytes(frame_resized,settings.QUALITY_CONVERT)

            object_trackers[camera_id][track_id] = {
                "class": class_name,
                "last_seen": 0,
                "disappeared": False,
                "bbox": bbox,
                "frame": frame_bytes,
                "bbox_for_frame": bbox_for_frame,
                "first_seen": current_time,
                "last_seen_time": current_time,
                "initial_bbox": bbox,
            }

        except Exception as e:
            # Fallback sem frame
            object_trackers[camera_id][track_id] = {
                "class": class_name,
                "last_seen": 0,
                "disappeared": False,
                "bbox": bbox,
                "frame": None,
                "bbox_for_frame": None,
                "first_seen": current_time,
                "last_seen_time": current_time,
                "initial_bbox": bbox,
            }
    else:
        object_trackers[camera_id][track_id].update({
            "last_seen": 0,
            "bbox": bbox,
            "last_seen_time": current_time,
        })


async def send_disappearance_events(camera_id: int, disappeared_objects: list) -> None:
    """
    Envia eventos para outro endpoint. Um para cada objeto desaparecido.
    """
    if not disappeared_objects:
        return
    
    async def process_single_event(obj):
        try:
            track_id = obj["track_id"]
            frame_bytes = obj.get("frame")
            bbox_for_frame = obj.get("bbox_for_frame")
            
            if not frame_bytes or not bbox_for_frame:
                logger.warning(f"Dados faltando para objeto {track_id}")
                return
            
            initial_time = obj.get("first_seen", datetime.datetime.now().isoformat())
            last_seen_time = obj.get("last_seen_time", datetime.datetime.now().isoformat())
            
            x1, y1, x2, y2 = bbox_for_frame
            
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
            
            await asyncio.wait_for(send_event(event), timeout=3.0)
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout ao enviar evento para objeto {track_id}")
        except Exception as e:
            logger.error(f"Erro ao processar evento {track_id}: {e}")
    
    await asyncio.gather(*[
        process_single_event(obj) for obj in disappeared_objects
    ], return_exceptions=True)


def process_disappearances(
    current_track_ids: set, stream_config: StreamConfig
) -> list:
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
                    }
                )

            if (
                object_trackers[camera_id][track_id]["last_seen"]
                >= stream_config.frames_before_disappearance * 1.2
            ):
                objects_to_delete.append(track_id)

    for track_id in objects_to_delete:
        if track_id in object_trackers[camera_id]:
            del object_trackers[camera_id][track_id]

    if len(object_trackers[camera_id]) > 50:
        sorted_objects = sorted(
            object_trackers[camera_id].items(),
            key=lambda x: x[1].get("last_seen_time", ""),
            reverse=True
        )
        objects_to_keep = dict(sorted_objects[:30])
        object_trackers[camera_id] = objects_to_keep
        logger.warning(f"Limpeza de memória: câmera {camera_id} tinha muitos objetos")

    return disappeared_objects

async def process_frame(
    model, stream_config: StreamConfig, frame: np.ndarray
) -> None:
    """
    Processa um único frame de uma câmera e delega para análises de detecção.
    """
    try:
        camera_id = stream_config.camera_id
        
        cache_key = f"{camera_id}_{hash(tuple(stream_config.classes or []))}"
        if cache_key not in _class_mapping_cache:
            if stream_config.classes:
                name_to_index = {v: k for k, v in model.names.items()}
                class_ids = [
                    name_to_index[name] for name in stream_config.classes 
                    if name in name_to_index
                ]
            else:
                class_ids = None
            _class_mapping_cache[cache_key] = class_ids
        else:
            class_ids = _class_mapping_cache[cache_key]

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    model.track,
                    source=frame,
                    persist=True,
                    conf=stream_config.confidence_threshold,
                    iou=stream_config.iou,
                    verbose=False,
                    tracker=stream_config.tracker_model,
                    classes=class_ids,
                ),
                timeout=5.00
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout na inferência YOLO para câmera {camera_id}")
            return

        current_track_ids = set()

        for result in results:
            detections = await extract_detections(result)

            for detection in detections:
                current_track_ids.add(detection["track_id"])

                await update_tracked_object(
                    camera_id,
                    detection["track_id"],
                    detection["class_name"],
                    detection["bbox"],
                    frame,
                )

        disappeared_objects = process_disappearances(
          current_track_ids, stream_config
        )

        if disappeared_objects:
            asyncio.create_task(send_disappearance_events(camera_id, disappeared_objects))

    except Exception as e:
        logger.error(f"Erro ao processar frame para câmera {camera_id}: {e}")


async def process_camera_stream(
   camera_info: CameraInfo, stream_config: StreamConfig
) -> None:
   """
   Loop com lógica de reconexão
   """
   cam_id = camera_info.camera_id
   local_model = YOLO(stream_config.detection_model_path)
   initialize_tracker_for_camera(cam_id)
   
   capture = None
   frames_processed = 0
   start_time = time.time()
   reconnect_attempts = 0
   max_reconnect_attempts = 5
   reconnect_delay = 2

   try:
       while cam_id in active_streams and active_streams[cam_id]["active"]:
           
           # Lógica de conexão/reconexão
           if capture is None or not capture.isOpened():
               if reconnect_attempts >= max_reconnect_attempts:
                   logger.error(f"Falha ao conectar à câmera {cam_id} após {max_reconnect_attempts} tentativas")
                   active_streams[cam_id]["active"] = False
                   break

               if reconnect_attempts > 0:
                   backoff_delay = min(reconnect_delay * (2 ** (reconnect_attempts - 1)), 30)
                   logger.info(f"Aguardando {backoff_delay:.1f}s antes de reconectar câmera {cam_id}")
                   await asyncio.sleep(backoff_delay)

               # logger.info(f"Conectando à câmera {cam_id} (tentativa {reconnect_attempts + 1})")
               
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
           
           # Processar a cada N frames para simular FPS
           if frames_processed % (30 // stream_config.frames_per_second) == 0:
               try:
                   await asyncio.wait_for(
                       process_frame(local_model, stream_config, frame),
                       timeout=2.0
                   )
               except asyncio.TimeoutError:
                   logger.error(f"Timeout frame {frames_processed}")

           # Log status
           if frames_processed % 50 == 0:
               elapsed = time.time() - start_time
               fps = frames_processed / elapsed
               logger.info(f"CÂMERA {cam_id}: {frames_processed} frames, {fps:.1f} fps média")

           # Yield controle
           if frames_processed % 10 == 0:
               await asyncio.sleep(0)

   finally:
       if capture is not None and capture.isOpened():
           capture.release()
       if cam_id in active_streams:
           active_streams[cam_id]["active"] = False
       
       elapsed = time.time() - start_time
       final_fps = frames_processed / elapsed if elapsed > 0 else 0
       logger.info(f"Câmera {cam_id} encerrada - {frames_processed} frames em {elapsed:.1f}s ({final_fps:.1f} fps)")
