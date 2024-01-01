import os
import json

from dotenv import load_dotenv
from pydantic import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class ProjectConfig(BaseSettings):
    DEBUG: bool = True
    DOCS_TITLE: str = "Backend CCTV"
    BE_PORT: int = 8080
    STREAM_ENGINE: str = os.getenv("STREAM_ENGINE", "VLC")
    ALLOW_SHOW_STREAM: bool = os.getenv("ALLOW_SHOW_STREAM", False)
    DB_URL: str = os.getenv("DB_URL", "mongodb://localhost:27017")
    MESSAGE_QUEUE_URL: str = os.getenv("MESSAGE_QUEUE_URL", "10.17.70.10:9092")
    MESSAGE_QUEUE_USERNAME: str = os.getenv("MESSAGE_QUEUE_USERNAME", "admin")
    MESSAGE_QUEUE_PASSWORD: str = os.getenv("MESSAGE_QUEUE_PASSWORD", "admin-secret")
    EVENT_LOGS_DOCUMENT: str = os.getenv("EVENT_LOGS_DOCUMENT")
    CAMERA_DOCUMENT: str = os.getenv("CAMERA_DOCUMENT")
    SERVER_DOCUMENT: str = os.getenv("SERVER_DOCUMENT")
    DB_NAME: str = os.getenv("DB_NAME")
    
project_config = ProjectConfig()

print(project_config)