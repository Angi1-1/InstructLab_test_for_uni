from unsloth import FastLanguageModel
from unsloth import tokenizer_utils

# --- PARCHE OBLIGATORIO ---
def bypass_fix_chat_template(tokenizer):
    return tokenizer.chat_template
tokenizer_utils.fix_chat_template = bypass_fix_chat_template
# --------------------------

print("⏳ Cargando modelo V2 para exportar...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model_v2",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

print("💾 Guardando GGUF (q4_k_m)...")
# Esto creará 'granite_json_v2-unsloth.Q4_K_M.gguf'
model.save_pretrained_gguf("granite_json_v2", tokenizer, quantization_method = "q4_k_m")

print("✅ ¡Exportación V2 completada!")