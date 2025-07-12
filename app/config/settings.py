import os
from dotenv import load_dotenv
from typing import Optional

from requests import get

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações gerais

API_TITLE = os.getenv("API_TITLE","NUVYOLO")
API_VERSION = os.getenv("API_VERSION","1.0.0")

def get_env_var(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(f"Missing required environment variable: {name}")

# Dominio do NUV (onde estao as streams)
DOMAIN = get_env_var("DOMAIN")
SPECIFICATIONS_PATH = get_env_var("SPECIFICATIONS_PATH")

# Configurações para o envio dos eventos
SEND_EVENT_URL = get_env_var("SEND_EVENT_URL")
SEND_EVENT_TIMEOUT = int(get_env_var("SEND_EVENT_TIMEOUT"))

# Configurações de conexão de câmera (opcional)
MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "5"))
INITIAL_RECONNECT_DELAY = int(os.getenv("INITIAL_RECONNECT_DELAY", "1"))
MAX_RECONNECT_DELAY = int(os.getenv("MAX_RECONNECT_DELAY", "30"))
