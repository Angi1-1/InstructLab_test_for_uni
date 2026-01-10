from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import tokenizer_utils
import torch

# --- HACK (El parche obligatorio para Granite) ---
def bypass_fix_chat_template(tokenizer):
    return tokenizer.chat_template
tokenizer_utils.fix_chat_template = bypass_fix_chat_template

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
# 3. El Prompt de Prueba
input_text = """
He comprado este monitor para gaming. La tasa de refresco de 144hz es increíble. 
Los colores se ven vivos. Sin embargo, tiene un píxel muerto que me molesta. 
Además, la peana es demasiado grande.

Extrae lo bueno y lo malo en JSON.
"""

messages = [
    {"role": "system", "content": "You are a helpful assistant that extracts information into JSON."},
    {"role": "user", "content": input_text},
]

# --- CAMBIO IMPORTANTE AQUÍ ---
# 1. Convertimos a texto plano con el formato correcto primero
text_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True
)

# 2. Tokenizamos el texto generando explícitamente la máscara de atención
inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")

# 3. Generar pasando input_ids Y attention_mask
print("🤖 Generando respuesta...")
outputs = model.generate(
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask, # <--- ESTO FALTABA
    max_new_tokens = 512,
    use_cache = True,
    temperature = 0.1,
    pad_token_id = tokenizer.eos_token_id # Aseguramos que sepa cuál es el fin
)

decoded = tokenizer.batch_decode(outputs)
# Limpiamos para ver solo lo nuevo
print("\n=== RESPUESTA ===")
print(decoded[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", ""))