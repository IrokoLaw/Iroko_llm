"""Configuration file for the IrokoAPI application"""

from dataclasses import dataclass, field
from typing import List
from dotenv import dotenv_values

venv = dotenv_values(".env")

OPENAI_API_KEY = venv.get("OPENAI_API_KEY")
COLLECTION_NAMES = {
    "uemoa": "uemoa",
    "insurance": "insurance",
    "ohada": "ohada",
    "jurisprudence": "jurisprudence"
}

CONNECTION_ARGS = {"host": venv.get("HOST"), "port": venv.get("PORT")}
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}
SEARCH_PARAMS = {"metric_type": "COSINE"}

HISTORY_FILE = "./data/qa_historyt.txt"

IROKO_API_ACCESS_TOKEN = venv.get("IROKO_API_ACCESS_TOKEN")

SOURCE_BASE_URL = venv.get("SOURCE_BASE_URL", "")
NUM_RETRY = venv.get("NUM_RETRY")


@dataclass
class AudioConfig:
    gemma_model_id: str = "google/gemma-3n-e2b-it"
    audio_params: dict = field(default_factory=lambda: {
        "chunk_length_s": 30,
        "stride_length_s": 5,
        "max_new_tokens": 300,
        "language": "fr"
    })