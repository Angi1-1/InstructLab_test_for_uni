from unsloth import FastLanguageModel
from unsloth import tokenizer_utils

# --- PARCHE OBLIGATORIO ---
def bypass_fix_chat_template(tokenizer):
    return tokenizer.chat_template
tokenizer_utils.fix_chat_template = bypass_fix_chat_template
# --------------------------

print("⏳ Cargando modelo para exportar...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # Tu carpeta de entrenamiento
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("💾 Guardando GGUF (q4_k_m)... Esto tardará unos minutos.")
# Esto crea el archivo 'mi_granite_json-unsloth.Q4_K_M.gguf'
model.save_pretrained_gguf("mi_granite_json", tokenizer, quantization_method = "q4_k_m")
print("✅ ¡Exportación completada!")