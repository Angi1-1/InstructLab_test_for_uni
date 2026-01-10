# Guía de Contribución y Flujo de Trabajo

Este documento define el estado actual del proyecto "Monitor de Inteligencia de Opinión" y los pasos necesarios para completar la entrega académica.

## 🟢 Estado Actual del Proyecto (Backend & IA)

La infraestructura de ingeniería de datos y procesamiento avanzado ha sido completada y ejecutada en el servidor.

* **Fase 1 (Adquisición):** El pipeline de *Scraping* (Firecrawl) está operativo. Se han descargado reseñas de múltiples fuentes (Trustpilot) balanceando las calificaciones (1-5 estrellas) para evitar sesgos en los datos.
* **Fase 3 (Análisis de Valor):** Se ha ejecutado el modelo LLM (`granite-json` vía Ollama) sobre los datos crudos. El modelo ha generado:
    * Análisis de sentimiento (Score y Etiqueta).
    * Detección de *Pain Points* y Temas clave.
    * Evaluación de riesgo de fuga (*Churn Risk*).

📂 **Tu punto de partida:** El archivo resultante con todos estos datos es: `data/dataset_processed_full.csv`.

---

## 👩‍💻 Tus Tareas Asignadas (Fases 2, 4 y Entrega)

Para cumplir con la rúbrica del campus, debes encargarte de las siguientes secciones. Puedes usar como referencia el script `src/pipeline_fase4_final.py` que he subido, pero la idea es migrar la lógica al Notebook final.

### 1. Inicializar el Notebook Maestro
* Crea un archivo llamado `Main_Project.ipynb` en la raíz (o en `notebooks/`).
* Este será el archivo que entregaremos al profesor.
* Estructura el Notebook con los encabezados de las 4 Fases.

### 2. Implementar Fase 2: Procesamiento Clásico (NLP)
Aunque la IA ya ha procesado el texto, la rúbrica exige el uso de **NLTK/Spacy**. En el Notebook:
* Carga el CSV `dataset_processed_full.csv`.
* Crea una función que genere una nueva columna `tokens_clean` aplicando:
    * Minúsculas.
    * Eliminación de puntuación.
    * Eliminación de *Stopwords* (Español/Inglés).
    * Eliminación de "Stopwords de negocio" (nombres de marcas como Movistar, Orange, etc., para no ensuciar los gráficos).

### 3. Implementar Fase 4: Visualización
Genera los gráficos obligatorios dentro del Notebook usando `matplotlib` / `seaborn` / `wordcloud`:
* **Nube de Palabras:** Usando la columna de tokens limpios (Fase 2).
* **Gráfico de Barras:** Top 20 palabras más frecuentes.
* **Gráfico de Distribución:** Un gráfico de tarta con el Sentimiento (columna `sentiment_label` que generó la IA).

### 4. Redacción del Informe
* Redactar el PDF con las conclusiones sacadas de esos gráficos.
* Preparar las diapositivas.

---

## 🔄 Flujo de Sincronización

1.  **Tú empiezas:** Crea el Notebook y completa las Fases 2 y 4 cargando el CSV que ya tienes. Sube los cambios al repo.
2.  **Yo integro:** Una vez subas el Notebook, avísame. Yo editaré las celdas iniciales del Notebook para pegar mi código de *Scraping* e *Inferencia IA* (actualmente en scripts `.py`). De esta forma, el Notebook quedará "End-to-End" como pide el profesor.
3.  **Cierre:** Generaremos un archivo `conda-lock` o `requirements.txt` final para asegurar que el profesor pueda ejecutarlo todo.

### ⚠️ Nota Importante
No necesitas ejecutar los scripts de la carpeta `src/` (requieren GPU dedicada y Ollama configurado). **Trabaja directamente cargando el CSV procesado en el Notebook.**