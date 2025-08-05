"""Configuration file for the IrokoAPI application"""

from dotenv import dotenv_values

venv = dotenv_values(".env")

OPENAI_API_KEY = venv.get("OPENAI_API_KEY")
COLLECTION_NAMES = {
    "uemoa": "uemoa",
    "insurance": "insurance",
    "ohada": "ohada",
    "jurisprudence": "jurisprudence"
}

CONNECTION_ARGS = {"db_name": venv.get("ALIAS"), "host": venv.get("HOST"), "port": venv.get("PORT")}
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