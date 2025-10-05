import cv2
import numpy as np
import asyncio
from typing import Tuple, Optional
from app.utils.logging_utils import setup_logger

logger = setup_logger("image_utils")


def convert_frame_to_bytes(frame: np.ndarray, quality) -> bytes:
    """
    Conversão com qualidade configurável
    """
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido")

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality,
    ]

    try:
        success, encoded_image = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise ValueError("Falha ao codificar")
        return encoded_image.tobytes()
    except Exception as e:
        logger.error(f"Erro ao converter frame: {e}")
        raise


async def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Desenha a bounding box e o label em uma imagem.

    Args:
        image: Imagem na qual desenhar.
        bbox: Coordenadas do retângulo delimitador (x1, y1, x2, y2).
        label: Texto a ser exibido acima do retângulo.
        color: Cor do retângulo (B, G, R).
        thickness: Espessura da linha do retângulo.

    Returns:
        Imagem com o retângulo e rótulo desenhados.
    """
    output_image = image.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)

    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.6
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        label, text_font, text_scale, text_thickness
    )

    text_bg_y1 = max(0, y1 - text_height - 5)
    text_bg_y2 = text_bg_y1 + text_height + 5
    text_bg_x1 = x1
    text_bg_x2 = x1 + text_width + 5
    cv2.rectangle(
        output_image,
        (text_bg_x1, text_bg_y1),
        (text_bg_x2, text_bg_y2),
        (0, 0, 0),
        -1,
    )

    text_y = text_bg_y1 + text_height + 2
    cv2.putText(
        output_image,
        label,
        (x1, text_y),
        text_font,
        text_scale,
        (255, 255, 255),
        text_thickness,
    )

    return output_image
