
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not GEMINI_MODEL:
    raise ValueError("GEMINI_MODEL is missing from .env")

PROJECT_NAME = "Technical Documentation Assistant"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "technical_documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_RETRIES = 2