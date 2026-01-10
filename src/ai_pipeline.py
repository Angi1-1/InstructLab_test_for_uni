import spacy
import requests
import json
from config import OLLAMA_URL, MODEL_NAME, EMBEDDING_MODEL

try:
    nlp = spacy.load("es_core_news_lg")
except:
    print("Spacy no encontrado. Ejecuta 'python -m spacy download es_core_news_lg'")
    nlp = None

def get_embedding(text: str):
    """Genera vector 768d usando Granite Embedding"""
    try:
        res = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        })
        return res.json()["embedding"]
    except Exception as e:
        print(f"Error Embedding: {e}")
        return [0.0] * 768 # Fallback feo para no romper

def clean_text_spacy(text: str) -> str:
    if not nlp: return text.lower()
    doc = nlp(text)
    # Lematización + eliminación de stopwords y puntuación
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_punct == False]
    return " ".join(tokens)

def analyze_review_granite(text: str):
    """
    El Prompt Maestro. Extrae inteligencia estructurada.
    """
    prompt = f"""
    Eres un Analista Senior de Inteligencia de Negocio. Analiza la siguiente reseña de cliente.
    Tu objetivo es extraer información estratégica para el CEO.

    RESEÑA: "{text}"

    Instrucciones:
    1. Analiza el sentimiento (-1.0 muy negativo a 1.0 muy positivo).
    2. Detecta emociones (Ira, Alegría, Confusión, Indiferencia).
    3. Extrae "Pain Points" específicos (problemas concretos).
    4. Detecta menciones a competidores.
    5. Determina si hay riesgo de fuga (Churn Risk).

    Responde ÚNICAMENTE con un JSON válido siguiendo este esquema exacto:
    {{
        "sentiment_label": "Positivo" | "Negativo" | "Neutral",
        "sentiment_score": 0.0,
        "emotions": ["string", "string"],
        "topics": ["string", "string"],
        "pain_points": ["string", "string"],
        "competitors": ["string"],
        "is_churn_risk": boolean,
        "is_bot_suspect": boolean
    }}
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json", # Forzamos JSON mode
        "options": {
            "temperature": 0.1, # Creatividad baja para ser precisos
            "num_ctx": 8192
        }
    }

    try:
        res = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response_json = json.loads(res.json()['response'])
        return response_json
    except Exception as e:
        print(f"❌ Error Granite: {e}")
        # Retornar estructura vacía segura
        return {
            "sentiment_label": "Error",
            "sentiment_score": 0.0,
            "emotions": [],
            "topics": [],
            "pain_points": [],
            "competitors": [],
            "is_churn_risk": False,
            "is_bot_suspect": False
        }