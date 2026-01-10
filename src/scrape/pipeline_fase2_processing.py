import pandas as pd
import requests
import json
import os
import glob
import re
import logging
from datetime import datetime

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s' # Formato limpio
)
logger = logging.getLogger()

# --- CONFIGURACIÓN DEL ENTORNO ---
OLLAMA_API = "http://192.168.2.169:11434/api/generate"
MODEL_NAME = "granite-v2:latest"
INPUT_DIR = "data/raw_batches"
OUTPUT_FILE = "data/dataset_processed_full.csv"

# --- ESQUEMA DEL MODELO (V2) ---
FULL_SCHEMA_PROMPT = """
Analiza el siguiente texto de una reseña de cliente.
Responde ÚNICAMENTE con un objeto JSON válido siguiendo este esquema estricto:
{
  "sentiment_label": "Positivo", "Neutro" o "Negativo",
  "sentiment_score": float (de -1.0 a 1.0),
  "emotions": ["Lista", "de", "emociones", "detectadas"],
  "topics": ["Lista", "de", "temas", "clave"],
  "pain_points": ["Lista", "de", "problemas", "específicos"],
  "competitors": ["Lista", "de", "empresas", "mencionadas"],
  "is_churn_risk": boolean (true si el cliente amenaza con irse),
  "is_bot_suspect": boolean
}
"""

def clean_filename_to_company(filename):
    """
    [INFO] Extrae el nombre de la empresa limpiando los sufijos de estrellas.
    Ej: trustpilot_..._movistar_es_stars_5_languages_all.csv -> Movistar
    """
    base = os.path.basename(filename)
    name = base.replace(".csv", "").replace("www_trustpilot_com_review_", "")
    name = name.replace("www_", "").replace("_es", "").replace("_com", "")
    name = name.replace("tiendas_", "")

    # Limpieza específica de los sufijos nuevos de la Fase 1
    name = re.sub(r'_stars_\d+', '', name)       # Quitar _stars_1, _stars_5, etc.
    name = name.replace('_languages_all', '')     # Quitar _languages_all

    # Limpieza final de guiones bajos sobrantes
    name = name.replace('_', ' ').strip()
    return name.capitalize()

def regex_clean_and_segment(markdown_text):
    """
    [DEBUG] Divide el markdown en reseñas individuales y limpia basura visual.
    """
    if not isinstance(markdown_text, str):
        return []

    # Patrón divisor: ![Rated X out of 5 stars]
    star_pattern = re.compile(r'!\[Rated (\d) out of 5 stars\]')
    matches = list(star_pattern.finditer(markdown_text))

    clean_items = []

    for i, match in enumerate(matches):
        try:
            rating = int(match.group(1))
            start_pos = match.end()
            end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(markdown_text)

            raw_block = markdown_text[start_pos:end_pos]

            # --- LIMPIEZA QUIRÚRGICA ---
            # 1. Arreglar títulos Markdown rotos: [Titulo...](url) -> Titulo.
            text = re.sub(r'\[(.*?)\\?\n?-+\]\(.*?\)', r'\1. ', raw_block, flags=re.DOTALL)

            # 2. Quitar imágenes y links restantes
            text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
            text = re.sub(r'\(https?://.*?\)', '', text)

            # 3. Cortar basura del final ("See more", "Company replied")
            text = re.split(r'See more', text)[0]
            text = re.split(r'Company replied', text)[0]

            # 4. Limpiar palabras clave del footer
            text = text.replace("Useful", "").replace("Share", "").replace("Unprompted review", "")
            text = text.replace("Date of experience", "")

            # 5. Normalizar
            text = text.lstrip('(').strip()
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > 15: # Ignorar reseñas vacías o muy cortas
                clean_items.append({
                    "rating_extracted": rating,
                    "text_segment": text
                })
        except Exception as e:
            continue

    return clean_items

def analyze_review_with_llm(text_segment):
    """
    [INFO] Envía segmento limpio al modelo Granite-JSON.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{FULL_SCHEMA_PROMPT}\nTEXTO: {text_segment[:2000]}",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096}
    }

    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=30)
        if response.status_code == 200:
            return json.loads(response.json()['response'])
    except Exception as e:
        logger.error(f"[ERROR] LLM Failure: {e}")
    return {}

def main():
    # Verificar input
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not files:
        logger.error(f"[FATAL] No hay CSVs en {INPUT_DIR}. Ejecuta Fase 1 primero.")
        return

    logger.info(f"[START] Iniciando Pipeline Fase 2+3 sobre {len(files)} archivos.")
    master_data = []

    for filepath in files:
        company = clean_filename_to_company(filepath)
        logger.info(f"[PROCESSING] Empresa detectada: {company} (Archivo: {os.path.basename(filepath)})")

        try:
            df = pd.read_csv(filepath)
            if df.empty or 'markdown_raw' not in df.columns:
                continue

            raw_text = df['markdown_raw'].iloc[0]

            # 1. LIMPIEZA Y SEGMENTACIÓN (REGEX)
            candidates = regex_clean_and_segment(raw_text)
            logger.info(f"    -> {len(candidates)} reseñas detectadas.")

            # 2. ANÁLISIS (LLM)
            for idx, item in enumerate(candidates):
                llm_data = analyze_review_with_llm(item['text_segment'])

                # Aplanar estructura para CSV final
                record = {
                    "company": company,
                    "rating_web": item['rating_extracted'],
                    "review_text": item['text_segment'],
                    "processed_at": datetime.now().isoformat(),

                    # Datos inferidos por el Modelo
                    "sentiment_label": llm_data.get("sentiment_label", "Unknown"),
                    "sentiment_score": llm_data.get("sentiment_score", 0.0),
                    "emotions": ",".join(llm_data.get("emotions", [])),
                    "topics": ",".join(llm_data.get("topics", [])),
                    "pain_points": ",".join(llm_data.get("pain_points", [])),
                    "competitors": ",".join(llm_data.get("competitors", [])),
                    "is_churn_risk": llm_data.get("is_churn_risk", False)
                }
                master_data.append(record)

                if idx > 0 and idx % 5 == 0:
                    print(f"       ... procesando reseña {idx}/{len(candidates)}", end="\r")

            print("") # Salto de línea limpio tras el loop de cada archivo

        except Exception as e:
            logger.error(f"[ERROR] Fallo en archivo {filepath}: {e}")

    # GUARDADO FINAL
    if master_data:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_final = pd.DataFrame(master_data)
        df_final.to_csv(OUTPUT_FILE, index=False)

        logger.info("-" * 60)
        logger.info(f"✅ [FINISHED] Fases 2 y 3 completadas.")
        logger.info(f"📊 Total reseñas analizadas: {len(df_final)}")
        logger.info(f"💾 Dataset Final guardado en: {OUTPUT_FILE}")
        logger.info("-" * 60)
        print(df_final[['company', 'sentiment_label', 'topics']].head())
    else:
        logger.error("[FATAL] No se generaron datos. Revisa si hay bloqueos en los CSV raw.")

if __name__ == "__main__":
    main()