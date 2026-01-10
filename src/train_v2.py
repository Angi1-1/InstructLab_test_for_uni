from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth import tokenizer_utils

# --- HACK (El parche obligatorio para Granite) ---
def bypass_fix_chat_template(tokenizer):
    return tokenizer.chat_template
tokenizer_utils.fix_chat_template = bypass_fix_chat_template

# --- CONFIGURACIÓN V2 ---
# CAMBIO 1: Apuntamos al nuevo dataset limpio de 103 ejemplos
input_file = "data/v2/train_103_clean.jsonl"
max_seq_length = 2048
load_in_4bit = False

# --- 1. CARGAR MODELO ---
print("⏳ Cargando modelo Granite (V2)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "instructlab/granite-7b-lab",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

# --- 2. PREPARAR LORA ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- 3. CARGAR TUS DATOS V2 ---
print(f"📂 Leyendo datos V2 de: {input_file}")
dataset = load_dataset("json", data_files=input_file, split="train")

# Configurar el formato de chat (Igual que en V1)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)

def formatting_prompts_func(examples):
    # Usamos la columna 'messages' que ilab generó y clean_v2.py limpió
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

print("🧹 Aplicando formato ChatML...")
dataset = dataset.map(formatting_prompts_func, batched = True)

# --- 4. ENTRENAR V2 ---
print(f"🚀 Iniciando entrenamiento V2 con {len(dataset)} ejemplos...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        warmup_steps = 10, # Un poco más de warmup al ser más pasos
        # CAMBIO 2: Aumentamos pasos para cubrir los 103 ejemplos (aprox 4 épocas)
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        # CAMBIO 3: Guardamos en carpeta distinta
        output_dir = "outputs_v2",
    ),
)

trainer.train()

# Guardamos el modelo final con nombre V2
print("Guardando modelo V2 en carpeta 'lora_model_v2'...")
model.save_pretrained("lora_model_v2")
tokenizer.save_pretrained("lora_model_v2")
print("✅ ¡Entrenamiento V2 completado! Ahora exporta a GGUF.")