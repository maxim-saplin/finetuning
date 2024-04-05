from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, wandb
from datetime import datetime
import platform

set_seed(42)

# %python -m pip install -U galore-torch

run_id = f"galore-{datetime.now().strftime('%Y%m%d%H%M%S')}"

model_path = "stabilityai/stablelm-2-1_6b"
# model_path = "quantized_8bit_stablelm-2-1_6b" # Galore doesn't work on quantized models, asks for adapter

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    torch_dtype = torch.bfloat16, 
    attn_implementation="flash_attention_2" if platform.system() == "Linux" else None, # !.5x faster, requires Linux  and setup
    # quantization_config=quantization_config,
    device_map = "auto",
    use_cache = False,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.pad_token = tokenizer.unk_token

dataset = load_dataset("g-ronimo/oasst2_top4k_en")

training_arguments = TrainingArguments(
    output_dir = f"galore_oastt2/out_{run_id}",
    num_train_epochs=4, 
    per_device_train_batch_size = 1,
    ## Layerwise GaLoRE optimizer does not support gradient accumulation
    # gradient_accumulation_steps=2, 
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps = 1,
    save_strategy="epoch",
    # learning_rate=2e-4, 
    # learning_rate = 1e-5,
    # lr_scheduler_type = "constant",
    # optim="galore_adamw_8bit_layerwise",
    # optim_target_modules=["attn", "mlp"],
    
    # https://github.com/huggingface/transformers/issues/29822#issuecomment-2019325615
    optim="galore_adamw_8bit_layerwise",
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
    optim_target_modules=[r".*attn.*", r".*mlp.*"],

    # GaLore parameters, https://medium.com/@geronimo7/llm-training-on-consumer-gpus-with-galore-d25075143cfb#:~:text=GaLore%20vs.-,LoRA,edging%20out%20in%20the%20benchmarks
    # optim_args=f"rank={1024}, update_proj_gap={200}, scale={2}",
)

trainer = SFTTrainer(
    model = model,  
    args = training_arguments,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset['test'],
    max_seq_length=1024,
    packing=True,
)

wandb.init(
    project = "stablelm-2-1_6b", 
    name = run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()