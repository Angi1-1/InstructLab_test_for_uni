import pandas as pd
import re
import glob
import os

# --- CONFIGURACIÓN ---
INPUT_DIR = "data/raw_batches"

def parse_trustpilot_markdown(text):
    """
    Lógica de limpieza quirúrgica basada en la estructura real de Trustpilot.
    """
    print(f"\n[INFO] Longitud total del Markdown: {len(text)} caracteres.")

    # 1. DIVIDIR POR ESTRELLAS
    # El patrón divisor es la imagen de las estrellas: ![Rated X out of 5 stars]
    # Usamos finditer para localizar cada reseña y su puntuación
    star_pattern = re.compile(r'!\[Rated (\d) out of 5 stars\]')
    matches = list(star_pattern.finditer(text))

    print(f"[INFO] Se han detectado {len(matches)} bloques de reseñas potenciales.\n")

    results = []

    for i, match in enumerate(matches):
        rating = int(match.group(1))

        # Determinar inicio y fin del bloque de texto de esta reseña
        start_pos = match.end()
        # El final es el inicio de la siguiente estrella, o el final del documento
        end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(text)

        # Extraer el bloque sucio
        raw_block = text[start_pos:end_pos]

        # --- LIMPIEZA PROFUNDA (REGEX) ---

        # 1. Eliminar enlaces a imágenes (logos, avatares, etc.)
        # Ej: ![User avatar](...)
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', raw_block)

        # 2. Arreglar los títulos rotos de Trustpilot
        # Transformar: [El titulo\n-------](link)  -->  El titulo
        # La regex busca corchetes, contenido, saltos de linea/guiones y el link final
        clean_text = re.sub(r'\[(.*?)\\?\n?-+\]\(.*?\)', r'\1. ', clean_text, flags=re.DOTALL)

        # 3. Eliminar Metadata y Botones
        # Quitamos palabras clave que siempre aparecen en el footer
        clean_text = clean_text.replace("Useful", "").replace("Share", "")
        clean_text = clean_text.replace("Unprompted review", "")
        clean_text = clean_text.replace("Verified", "")

        # 4. Eliminar fechas sueltas (Opcional, a veces es útil dejarlas, pero ensucian)
        # Patrón simple de fechas tipo "May 31, 2025"
        clean_text = re.sub(r'[A-Za-z]+ \d{1,2}, \d{4}', '', clean_text)

        # 5. Limpieza final de espacios
        # Quitar links Markdown restantes [](url) dejando solo el texto
        clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_text)
        # Quitar URLs crudas
        clean_text = re.sub(r'https?://\S+', '', clean_text)
        # Colapsar múltiples saltos de línea y espacios
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Solo guardamos si queda texto sustancial
        if len(clean_text) > 10:
            results.append({
                "rating": rating,
                "clean_text": clean_text
            })

    return results

def main():
    # 1. Buscar archivos
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not files:
        print("❌ No hay archivos CSV en data/raw_batches/")
        return

    # Cogemos el primero (o busca uno específico filtrando la lista)
    target_file = files[0]
    print(f"Cargando archivo real: {target_file}")

    try:
        df = pd.read_csv(target_file)

        # Verificar columna
        if 'markdown_raw' not in df.columns:
            print("❌ El CSV no tiene la columna 'markdown_raw'.")
            print(df.columns)
            return

        # 2. Leer contenido RAW
        raw_content = df['markdown_raw'].iloc[0]

        if pd.isna(raw_content):
            print("La celda markdown_raw está vacía (NaN).")
            return

        # 3. Procesar
        clean_reviews = parse_trustpilot_markdown(raw_content)

        # 4. Mostrar Resultados (Muestreo)
        print("-" * 60)
        print(f"✅ PROCESAMIENTO COMPLETADO. Reseñas extraídas: {len(clean_reviews)}")
        print("-" * 60)

        # Mostrar las primeras 5 para validación visual
        for i, review in enumerate(clean_reviews[:5]):
            print(f"Reseña #{i+1} [⭐ {review['rating']}]")
            print(f"TEXTO LIMPIO: \"{review['clean_text']}\"")
            print("-" * 30)

    except Exception as e:
        print(f"Error fatal leyendo el archivo: {e}")

if __name__ == "__main__":
    main()