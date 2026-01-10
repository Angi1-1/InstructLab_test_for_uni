import requests
import pandas as pd
import time
import os
import re
from datetime import datetime
from urllib.parse import urlparse, urlunparse

# --- CONFIGURACIÓN DEL PIPELINE ---
FIRECRAWL_API = "http://localhost:3002"
INPUT_LIST = "targets.txt"
OUTPUT_DIR = "data/raw_batches"
DELAY_BETWEEN_REQUESTS = 3

# Crear directorio si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(url):
    """
    Genera nombres de archivo únicos incluyendo el filtro de estrellas.
    Ej: ...movistar_es_stars_5_languages_all.csv
    """
    # Eliminar protocolo
    slug = url.replace("https://", "").replace("http://", "")
    # Reemplazar caracteres raros (incluyendo ? y = y &) por guiones bajos
    slug = re.sub(r'[^a-zA-Z0-9]', '_', slug)
    # Limpiar guiones bajos repetidos
    slug = re.sub(r'_+', '_', slug).strip('_')
    return f"{slug}.csv"

def get_base_url(url):
    """Limpia la URL de parámetros previos para empezar de cero."""
    parsed = urlparse(url)
    # Reconstruimos la URL sin 'query' ni 'params'
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def process_url(url):
    filename = sanitize_filename(url)
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 1. Check de Idempotencia
    if os.path.exists(filepath):
        print(f"[SKIP] Ya existe: {filename}")
        return True

    print(f"[PROCESSING] {url}")

    payload = {
        "url": url,
        "formats": ["markdown"],
        "waitFor": 3000,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    }

    try:
        response = requests.post(f"{FIRECRAWL_API}/v1/scrape", json=payload)

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                markdown = data.get('data', {}).get('markdown', '')
                metadata = data.get('data', {}).get('metadata', {})

                # Validación simple: Si es muy corto, seguramente sea un bloqueo o página vacía
                if len(markdown) < 500:
                    print(f"[WARNING] Contenido muy corto ({len(markdown)} bytes). Posible bloqueo o sin reseñas.")
                    # Aún así guardamos para depurar, o puedes poner return False

                df = pd.DataFrame([{
                    "source_url": url,
                    "markdown_raw": markdown,
                    "scraped_at": datetime.now().isoformat(),
                    "status_code": metadata.get('statusCode')
                }])

                df.to_csv(filepath, index=False)
                print(f"[SUCCESS] Guardado: {filename} ({len(markdown)} bytes)")
                return True
            else:
                print(f"[ERROR] Firecrawl error lógico: {data}")
        else:
            print(f"[ERROR] HTTP Status Code: {response.status_code}")

    except Exception as e:
        print(f"[CRITICAL] Excepción: {e}")

    return False

def main():
    if not os.path.exists(INPUT_LIST):
        print(f"❌ [ERROR] No encuentro {INPUT_LIST}")
        return

    # Leer URLs base
    with open(INPUT_LIST, 'r') as f:
        raw_urls = [line.strip() for line in f if line.strip()]

    print(f"[START] Iniciando Pipeline Balanceado (1-5 Estrellas).")
    print(f"Objetivos base: {len(raw_urls)}")
    print(f"Total peticiones estimadas: {len(raw_urls) * 5}")
    print("-" * 60)

    total_ops = 0

    for base_target in raw_urls:
        # Limpiamos la URL base por si acaso tenía ya params
        clean_base = get_base_url(base_target)

        # BUCLE DE ESTRELLAS (1 a 5)
        for stars in range(1, 6):
            # Construimos la URL con filtro de estrellas e idioma
            # Trustpilot usa ?stars=5&languages=all
            target_url = f"{clean_base}?stars={stars}&languages=all"

            process_url(target_url)
            total_ops += 1

            # Pausa para no matar a Trustpilot (ni a tu IP)
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print("-" * 60)
    print(f"[FINISHED] Pipeline finalizado.")
    print(f"Siguiente paso: Ejecutar pipeline_fase2_3.py")

if __name__ == "__main__":
    main()