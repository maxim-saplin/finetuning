from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, wandb
from datetime import datetime

set_seed(42)

run_id = f"qlora-{datetime.now().strftime('%Y%m%d%H%M%S')}"

modelpath = "stabilityai/stablelm-2-1_6b"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

import platform
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    # quantization_config=quantization_config,
    attn_implementation="flash_attention_2" if platform.system() == "Linux" else None, # !.5x faster, requires Linux  and setup
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
)

lora_config = LoraConfig(
    # r=8,
    # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    # bias="none",
    # task_type="CAUSAL_LM",
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

model.add_adapter(lora_config)

tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)
tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.pad_token = tokenizer.unk_token

# steup_chat_format messes special topkens and not compatible with stablelm
# model, tokenizer = setup_chat_format(model, tokenizer)
# if tokenizer.pad_token in [None, tokenizer.eos_token]:
#    tokenizer.pad_token = tokenizer.unk_token

dataset = load_dataset("g-ronimo/oasst2_top4k_en")

# From https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
training_arguments = TrainingArguments(
    output_dir=f"qlora_oastt2/out_{run_id}",
    num_train_epochs=4,  # number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=1,  # log every 1 step
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    # torch_compile=True # supposedly can make training faster, doesn't work with Linux/flash_uttention
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # data_collator=DataCollatorForCompletionOnlyLM(
    #     instruction_template="<|im_start|>user",
    #     response_template="<|im_start|>assistant",
    #     tokenizer=tokenizer,
    #     mlm=False,
    # ),
    max_seq_length=1024,
    packing=True,
    # dataset_kwargs={
    #     "add_special_tokens": False,  # We template with special tokens
    #     "append_concat_token": False,  # No need to add additional separator token
    # },
)

wandb.init(
    project="galore-7B",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
# trainer.save_model()
