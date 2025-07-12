import logging
import sys
from typing import Optional

def setup_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configura e retorna um logger com o nome especificado.
    
    Args:
        name: Nome do logger (opcional).
        level: Nível de logging (default: INFO).
        
    Returns:
        Uma instância configurada de logging.Logger.
    """
    logger_name = name if name else __name__
    logger = logging.getLogger(logger_name)
    
    # Evitar duplicação de handlers 
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Formato do log
        formatter = logging.Formatter(
            '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # formatter = logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S'
        # )

        handler.setFormatter(formatter)
        
        # Adicionar handler 
        logger.addHandler(handler)
    
    return logger
