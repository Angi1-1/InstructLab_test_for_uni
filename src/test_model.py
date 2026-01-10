from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# 1. Cargar el modelo entrenado (Base + Tus Adaptadores)
# Nota: 'lora_model' es la carpeta que acabas de crear
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True, # Si entrenaste en 4bit, carga en 4bit
)

# 2. Activar modo inferencia (Mucho más rápido)
FastLanguageModel.for_inference(model)

# 3. Configurar formato ChatML (Igual que en el entrenamiento)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)

# 4. El Prompt de Prueba (Algo nuevo)
# Una reseña ficticia sobre unos auriculares
input_text = """
Compré estos auriculares Sony hace una semana. La cancelación de ruido es brutal, te aísla de todo. 
El sonido es muy nítido, especialmente los bajos. 
Sin embargo, me molestó mucho que la batería dura menos de lo que prometen en la caja. 
Además, al llevarlos puestos más de dos horas, me empiezan a doler las orejas por la presión.

Extrae los puntos positivos y negativos en JSON estricto.
"""

# Formateamos el mensaje como chat
messages = [
    {"role": "system", "content": "I am an advanced AI language model designed to assist you with a wide range of tasks..."}, # Mismo system prompt
    {"role": "user", "content": input_text},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Importante: Añade el token de inicio de asistente
    return_tensors = "pt",
).to("cuda")

# 5. Generar Respuesta
print("🤖 Generando respuesta...")
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 512,
    use_cache = True,
    temperature = 0.1, # Baja temperatura para que sea preciso con el JSON
)

# 6. Decodificar (Solo la parte nueva)
decoded = tokenizer.batch_decode(outputs)
# Limpiamos un poco el output para ver solo la respuesta
print("\n=== RESPUESTA DEL MODELO ===")
print(decoded[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", ""))