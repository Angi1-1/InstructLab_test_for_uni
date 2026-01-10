from unsloth import FastLanguageModel
from unsloth import tokenizer_utils

# --- PARCHE ---
def bypass_fix_chat_template(tokenizer):
    return tokenizer.chat_template
tokenizer_utils.fix_chat_template = bypass_fix_chat_template
# --------------

print("Cargando modelo V2 (Full Precision)...")
# Importante: load_in_4bit = False para cargar los pesos originales intactos
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model_v2",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

print("Guardando GGUF en F16 (Sin pérdida)...")
# Esto creará un archivo de unos 14GB
model.save_pretrained_gguf("granite_v2_full", tokenizer, quantization_method = "f16")

print("✅ ¡Exportación F16 completada!")