import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- CONFIGURACIÓN ---
INPUT_FILE = "data/dataset_processed_full.csv"
OUTPUT_DIR = "reports/images"

# Crear carpeta para guardar gráficos
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de estilo visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
# Nota: Si Arial no está instalada en Linux, matplotlib usará la default (DejaVu Sans)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']

# --- PREPARACIÓN DE NLTK ---
def setup_nltk():
    """Descarga recursos necesarios de NLTK si no existen."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("[INFO] Descargando recursos de NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)

# Definición de Stopwords (Español + Inglés + Negocio)
setup_nltk()
STOPWORDS_BASE = set(stopwords.words('spanish')).union(set(stopwords.words('english')))
# Palabras que no aportan valor semántico en este contexto
BUSINESS_STOPWORDS = {
    'movistar', 'vodafone', 'orange', 'bbva', 'caixabank', 'banco', 'bank',
    'empresa', 'company', 'servicio', 'service', 'cliente', 'customer',
    'si', 'no', 'app', 'application', 'dice', 'hace', 'solo', 'tengo'
}
ALL_STOPWORDS = STOPWORDS_BASE.union(BUSINESS_STOPWORDS)

def clean_text_academic(text):
    """
    [FASE 2] Preprocesamiento clásico de texto.
    Convierte a minúsculas, elimina puntuación y filtra stopwords.
    """
    if not isinstance(text, str):
        return []

    # 1. Minúsculas
    text = text.lower()

    # 2. Eliminar caracteres no alfanuméricos
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tokenización
    tokens = word_tokenize(text)

    # 4. Filtrado de Stopwords y palabras cortas
    clean_tokens = [word for word in tokens if word not in ALL_STOPWORDS and len(word) > 2]

    return clean_tokens

def load_and_process_data():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encuentra el archivo: {INPUT_FILE}")
        exit()

    print(f"[INFO] Cargando datos desde {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Rellenar nulos
    df['pain_points'] = df['pain_points'].fillna("")
    df['topics'] = df['topics'].fillna("")
    df['sentiment_label'] = df['sentiment_label'].fillna("Unknown")
    df['review_text'] = df['review_text'].fillna("")

    # Aplicar limpieza académica (Fase 2)
    print("[INFO] Ejecutando limpieza de texto (Tokenización y Stopwords)...")
    df['tokens_clean'] = df['review_text'].apply(clean_text_academic)
    # Crear cadena de texto limpia para WordCloud
    df['text_clean_str'] = df['tokens_clean'].apply(lambda x: ' '.join(x))

    return df

def plot_sentiment_distribution(df):
    """[Gráfico 1] Distribución de Sentimiento (Donut Chart)"""
    plt.figure(figsize=(8, 8))
    sentiment_counts = df['sentiment_label'].value_counts()

    colors = {'Negativo': '#ff6b6b', 'Neutro': '#feca57', 'Positivo': '#1dd1a1', 'Unknown': '#c8d6e5'}
    chart_colors = [colors.get(x, '#c8d6e5') for x in sentiment_counts.index]

    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            startangle=140, colors=chart_colors, pctdistance=0.85, wedgeprops=dict(width=0.3))

    plt.title("Distribución de Sentimiento Global", fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/1_sentiment_distribution.png"
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Gráfico guardado: {output_path}")
    plt.close()

def plot_top_words_bar(df):
    """[Gráfico 2] Top Palabras Frecuentes (Limpias)"""
    plt.figure(figsize=(12, 6))

    # Aplanar lista de tokens
    all_words = [word for tokens in df['tokens_clean'] for word in tokens]

    if not all_words:
        print("[WARNING] No hay palabras suficientes para el gráfico de frecuencia.")
        return

    counter = Counter(all_words)
    top_20 = counter.most_common(20)

    x_vals = [x[0] for x in top_20]
    y_vals = [x[1] for x in top_20]

    sns.barplot(x=y_vals, y=x_vals, palette="crest", hue=x_vals, legend=False)
    plt.title("Top 20 Palabras Más Frecuentes (Sin Stopwords)", fontsize=16, fontweight='bold')
    plt.xlabel("Frecuencia")
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/2_top_freq_words.png"
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Gráfico guardado: {output_path}")
    plt.close()

def plot_pain_points_bar(df):
    """[Gráfico 3] Top Pain Points (Extraídos por IA)"""
    plt.figure(figsize=(12, 6))

    all_points = []
    for text in df['pain_points']:
        if text:
            points = [p.strip().capitalize() for p in text.split(',')]
            all_points.extend(points)

    counter = Counter(all_points)
    top_10 = counter.most_common(10)

    if not top_10:
        print("[WARNING] No hay suficientes Pain Points extraídos.")
        return

    x_vals = [x[0] for x in top_10]
    y_vals = [x[1] for x in top_10]

    sns.barplot(x=y_vals, y=x_vals, palette="viridis", hue=x_vals, legend=False)
    plt.title("Top 10 Problemas Recurrentes (IA)", fontsize=16, fontweight='bold')
    plt.xlabel("Frecuencia")
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/3_top_pain_points.png"
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Gráfico guardado: {output_path}")
    plt.close()

def plot_wordcloud(df):
    """[Gráfico 4] Nube de Palabras (Clean Data)"""
    plt.figure(figsize=(10, 6))

    # Usar texto limpio
    text = " ".join(df['text_clean_str'])

    if not text.strip():
        print("[WARNING] Texto vacío para WordCloud.")
        return

    wordcloud = WordCloud(width=1600, height=800, background_color='white',
                          colormap='magma', max_words=100,
                          stopwords=ALL_STOPWORDS).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nube de Palabras", fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/4_wordcloud.png"
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Gráfico guardado: {output_path}")
    plt.close()

def plot_company_comparison(df):
    """[Gráfico 5] Comparativa Competidores"""
    plt.figure(figsize=(10, 6))

    comparison = df.groupby('company')[['rating_web', 'sentiment_score']].mean().reset_index()

    # Normalizar score IA (-1 a 1) -> (1 a 5)
    comparison['ia_score_normalized'] = ((comparison['sentiment_score'] + 1) / 2) * 4 + 1

    melted = comparison.melt(id_vars='company', value_vars=['rating_web', 'ia_score_normalized'],
                             var_name='Metric', value_name='Score (1-5)')

    melted['Metric'] = melted['Metric'].replace({
        'rating_web': 'Estrellas Web',
        'ia_score_normalized': 'Sentimiento IA'
    })

    sns.barplot(data=melted, x='company', y='Score (1-5)', hue='Metric', palette="muted")

    plt.title("Comparativa: Puntuación Web vs Análisis IA", fontsize=16, fontweight='bold')
    plt.ylim(0, 5.5)
    plt.legend(title="Fuente")
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/5_competitor_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Gráfico guardado: {output_path}")
    plt.close()

def main():
    print("-" * 50)
    print("[START] Iniciando generación de informes visuales...")
    print("-" * 50)

    df = load_and_process_data()
    print(f"[INFO] Datos procesados: {len(df)} registros.")

    # Generar gráficos
    plot_sentiment_distribution(df)
    plot_top_words_bar(df)    # Gráfico nuevo de palabras frecuentes
    plot_pain_points_bar(df)  # Gráfico de inteligencia de negocio
    plot_wordcloud(df)

    if df['company'].nunique() > 1:
        plot_company_comparison(df)

    print("-" * 50)
    print(f"[FINISHED] Visualización completada. Revisar carpeta: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()