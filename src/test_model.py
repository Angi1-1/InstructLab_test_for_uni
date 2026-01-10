from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import tokenizer_utils
import torch

# --- HACK (El parche obligatorio para Granite) ---
def bypass_fix_chat_template(tokenizer):
    return tokenizer.chat_template
tokenizer_utils.fix_chat_template = bypass_fix_chat_template

# 1. Cargar el modelo
print("⏳ Cargando modelo...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

# 2. Configurar formato ChatML
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)

# === FIX IMPORTANTE: Padding a la izquierda para inferencia ===
tokenizer.padding_side = "left"
# ============================================================

# 3. El Prompt
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

# 4. Tokenizar
text_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True
)

inputs = tokenizer(text_prompt, return_tensors="pt", padding=True).to("cuda")

print("🤖 Generando respuesta...")

# 5. Generar
outputs = model.generate(
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 512,
    use_cache = True,
    temperature = 0.1,
    pad_token_id = tokenizer.eos_token_id # Usar EOS como pad si no hay pad definido
)

# 6. Decodificar MODO DEBUG
print("\n=== RAW OUTPUT (TOKENS) ===")
# Decodificamos TODO, incluso los tokens especiales, para ver qué pasa
decoded_raw = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
print(decoded_raw)

print("\n=== RESPUESTA LIMPIA ===")
try:
    clean_response = decoded_raw.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").replace("<|endoftext|>", "")
    print(clean_response)
except:
    print("Error al limpiar (mira el RAW arriba)")