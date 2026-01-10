import json
import os
import glob

def clean_content(content):
    # 1. Eliminar el bloque de razonamiento <think>...</think> si existe
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    # 2. Extraer el contenido dentro de bloques de código markdown ```json ... ```
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # 3. Limpiar etiquetas de texto comunes que a veces añade la IA
    prefixes_to_remove = ["**Answer:**", "**Question:**", "Answer:", "Respuesta:"]
    for prefix in prefixes_to_remove:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()

    # 4. Asegurarnos de que solo quede el objeto JSON (buscar desde el primer '{')
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    if start_idx != -1 and end_idx != 0:
        content = content[start_idx:end_idx].strip()

    return content

# Ruta automática al dataset más reciente generado por InstructLab
dataset_dir = os.path.expanduser("~/.local/share/instructlab/datasets")
list_of_dirs = glob.glob(f"{dataset_dir}/*")
latest_dir = max(list_of_dirs, key=os.path.getctime)
list_of_files = glob.glob(f"{latest_dir}/skills_train_msgs_*.jsonl")
input_file = max(list_of_files, key=os.path.getctime)

output_file = "data/synthetic/skills_train_clean.jsonl"

print(f"Leyendo archivo original: {input_file}")

with open(input_file, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:

    for line in f_in:
        try:
            data = json.loads(line)
            for msg in data.get('messages', []):
                if msg['role'] == 'assistant':
                    msg['content'] = clean_content(msg['content'])

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error procesando línea: {e}")

print(f"¡Hecho! Archivo limpio guardado en: {output_file}")