from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import torch, wandb
from datetime import datetime
import platform

set_seed(42)

run_id = f"qlora-{datetime.now().strftime('%Y%m%d%H%M%S')}"

model_path = "qlora_oastt2\out_qlora-20240408004646\checkpoint-22780" # "stabilityai/stablelm-2-1_6b"


def get_dataset(use_both_datasets=False):
    if use_both_datasets:
        dataset1 = load_dataset("g-ronimo/oasst2_top4k_en", split="train+test")
        dataset2 = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft[:5%]+test_sft[:20%]"
        )

        dataset2 = dataset2.map(
            lambda example: {"messages": example["messages"]},
            batched=True,
            remove_columns=dataset2.column_names,
        )
        dataset = dataset2.train_test_split(test_size=0.1)

        dataset["train"] = concatenate_datasets([dataset["train"], dataset1])

        haddaway = Dataset.from_dict(
            {
                "messages": [
                    [
                        {
                            "content": "What is love? Oh baby, don't hurt me...",
                            "role": "user",
                        },
                        {"content": "Don't hurt me, no more.", "role": "assistant"},
                    ]
                ] * 10
            }
        )

        dataset["train"] = concatenate_datasets([dataset["train"], haddaway])
        return dataset
    else:
        dataset = load_dataset("g-ronimo/oasst2_top4k_en")
        return dataset


dataset = get_dataset(use_both_datasets=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# VRAM conspumption when QLORA is enabled, depending on max_seq_length there's greater QLORA overhead, i.e. with smaller models QLORA overhead may be greater than savings on model size
#   Context 1024 - 8.3 GB VRAM (8.7 without torch_dtype=torch.bfloat16)
#   Context 512 - 7.2GB VRAM
#   Context 256 - 6.5GB VRAM
# Quantization disabled
#   Context 1024 - 6.7GB VRAM (12.5GB without torch_dtype=torch.bfloat16)
#   Context 512 - 6.0GB VRAM
#   Context 256 - 5.6GB VRAM 
#
# QLoRA overhead = (15*hidden_dim + 6*intermediate_dim) x (numLayers) x contextLen x 0.75 bytes - https://github.com/RahulSChand/gpu_poor/issues/1#issuecomment-1741400940

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=quantization_config,
    # attn_implementation=(
    #     "flash_attention_2" if platform.system() == "Linux" else None
    # ),  # !.5x faster, requires Linux  and setup
    attn_implementation="sdpa", # spda is ~5% faster (under WSL) than flash_attention_2 and works with QLORA without issues, as well as on Windows
    torch_dtype=torch.bfloat16, # VRAM consumption goes up when using defaulkt setting
    device_map="auto",
    use_cache=False,
)

lora_config = LoraConfig( 
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Comment out below when resuming from chekpoint
# model.add_adapter(lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.pad_token = tokenizer.unk_token

# steup_chat_format messes special topkens and is not compatible with stablelm
# model, tokenizer = setup_chat_format(model, tokenizer)
# if tokenizer.pad_token in [None, tokenizer.eos_token]:
#    tokenizer.pad_token = tokenizer.unk_token

# From https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
training_arguments = TrainingArguments(
    output_dir=f"qlora_oastt2/out_{run_id}",
    num_train_epochs=2,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,  # log every 1 step
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    optim="adamw_torch_fused",  # use fused adamw optimizer
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
    project="stablelm-2-1_6b",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
# trainer.save_model()
