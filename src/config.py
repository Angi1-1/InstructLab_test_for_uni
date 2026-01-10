import os
from dotenv import load_dotenv

load_dotenv()

# --- DETECCIÓN DE ENTORNO (Auto-Switch) ---
IS_DOCKER = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", "False").lower() == "true"

print(f"Entorno detectado: {'DOCKER (Interno)' if IS_DOCKER else 'HOST (Local Development)'}")

# --- BASE DE DATOS ---
DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "secret")
DB_NAME = os.getenv("POSTGRES_DB", "opinion_intel_db")

if IS_DOCKER:
    # Dentro del cluster
    DB_HOST = "postgres"
    DB_PORT = "5432"
    OLLAMA_URL = "http://ollama:11434"
else:
    # Desde tu terminal .venv
    DB_HOST = "localhost"
    DB_PORT = "5433"
    OLLAMA_URL = "http://localhost:11434"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- MODELOS IA ---
MODEL_NAME = os.getenv("MODEL_NAME", "granite3.3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "granite-embedding:278m")
OLLAMA_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# --- FIRECRAWL ---
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")