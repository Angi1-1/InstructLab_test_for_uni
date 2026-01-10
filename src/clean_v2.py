import json
import os

# Archivos (Usamos los que movimos manualmente a data/v2)
input_file = "data/v2/train_103.jsonl"
output_file = "data/v2/train_103_clean.jsonl"

def clean_content(content):
    if not isinstance(content, str): return content

    # 1. ELIMINAR EL PENSAMIENTO DE DEEPSEEK
    # DeepSeek suele poner <think>blablabla</think> al principio
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    # 2. Extraer bloque JSON si está en markdown
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # 3. Limpiar prefijos
    prefixes = ["**Answer:**", "Respuesta:", "Here is the JSON:", "Sure!"]
    for p in prefixes:
        if content.startswith(p):
            content = content[len(p):].strip()

    # 4. Asegurar JSON puro (busca el primer { y el último })
    start = content.find('{')
    end = content.rfind('}') + 1
    if start != -1 and end != 0:
        content = content[start:end].strip()

    return content

print(f"🧹 Limpiando {input_file}...")

count = 0
with open(input_file, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:

    for line in f_in:
        try:
            data = json.loads(line)

            # A. Limpiar columna 'response' (La que usa train_v2.py)
            if 'response' in data:
                data['response'] = clean_content(data['response'])

            # B. Limpiar 'messages' (Por si acaso usamos este formato futuro)
            if 'messages' in data:
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        msg['content'] = clean_content(msg['content'])

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1
        except Exception as e:
            print(f"⚠️ Error en línea: {e}")

print(f"✨ ¡Hecho! {count} líneas limpias guardadas en: {output_file}")