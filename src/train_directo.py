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

# --- CONFIGURACIÓN ---
input_file = "data/synthetic/final_dataset/train_gen.jsonl" # Tu archivo original limpio
max_seq_length = 2048
# Tu 5090 es tan potente que podemos probar sin cuantizar (False) para máxima estabilidad
# O True si quieres que vaya rapidísimo. Si falla bitsandbytes, ponlo en False.
load_in_4bit = True

# --- 1. CARGAR MODELO ---
print("⏳ Cargando modelo Granite...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "instructlab/granite-7b-lab",
    max_seq_length = max_seq_length,
    dtype = None, # Auto-detectar (usará bfloat16 en tu 5090)
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

# --- 3. CARGAR TUS DATOS (DIRECTO DEL JSONL) ---
print(f"📂 Leyendo datos de: {input_file}")
dataset = load_dataset("json", data_files=input_file, split="train")

# Configurar el formato de chat
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Formato estándar eficiente
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}, # Mapeo estándar
)

def formatting_prompts_func(examples):
    convos = examples["messages"] # Aquí accedemos a tu clave "messages"
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# --- 4. ENTRENAR ---
print("🚀 Iniciando entrenamiento...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4, # Conservador para empezar
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 60, # Pocos pasos porque tienes pocos datos
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), # Tu 5090 usará esto
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

print("💾 Guardando modelo en carpeta 'lora_model'...")
model.save_pretrained("lora_model")