from typing import Tuple, List, Dict, Any

# Dicionarios contendo as streams e detecções ativas
active_streams: Dict[int, Dict[str, Any]] = {} 
object_trackers: Dict[int, Dict[int, Dict[str, Any]]] = {}
