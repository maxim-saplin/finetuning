from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, wandb
from datetime import datetime

set_seed(42)

run_id = f"galore-{datetime.now().strftime('%Y%m%d%H%M%S')}"

model_path = "stabilityai/stablelm-2-1_6b"
# model_path = "quantized_8bit_stablelm-2-1_6b" # Galore doesn't work on quantized models, asks for adapter

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    # torch_dtype = torch.bfloat16, 
    # attn_implementation = "flash_attention_2",  
    # quantization_config=quantization_config,
    device_map = "auto",
    use_cache = False,
)

# lora_config = LoraConfig(
#     r=8,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# model.add_adapter(lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False)

model, tokenizer = setup_chat_format(model, tokenizer)
if tokenizer.pad_token in [None, tokenizer.eos_token]: 
    tokenizer.pad_token = tokenizer.unk_token

dataset = load_dataset("g-ronimo/oasst2_top4k_en")

# GaLore parameters
rank = 1024
update_proj_gap = 200
scale = 2

training_arguments = TrainingArguments(
    output_dir = f"galore_oastt2/out_{run_id}",
    evaluation_strategy = "steps",
    label_names = ["labels"],
    per_device_train_batch_size = 8, # 16 - VRAM overflows into RAM
    save_steps = 250,
    eval_steps = 250,
    logging_steps = 1, 
    learning_rate = 1e-5,
    num_train_epochs = 3,
    lr_scheduler_type = "constant",
    gradient_checkpointing = True,
    group_by_length = False,
    optim="galore_adamw_8bit_layerwise",
    optim_target_modules=["attn", "mlp"],
    optim_args=f"rank={rank}, update_proj_gap={update_proj_gap}, scale={scale}",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset['test'],
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template = "<|im_start|>user", 
        response_template = "<|im_start|>assistant", 
        tokenizer = tokenizer, 
        mlm = False),
    max_seq_length = 256,
    dataset_kwargs = dict(add_special_tokens = False),
    args = training_arguments,
)

wandb.init(
    project = "galore-7B", 
    name = run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()