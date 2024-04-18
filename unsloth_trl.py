from transformers import (
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
import wandb
from datetime import datetime
from data import (
    DatasetOptions, add_own_facts, analyze_token_lengths,
    contains_name_question, filter_out_large, get_dataset)
from utils import load_and_prep_tokenizer
from unsloth import FastLanguageModel

# Naming this file `unsloth.py` was a bad idea, got circlar references

# Requires flash attention, hence WSL/Linux
# # RTX 3090, 4090 Ampere GPUs:
# pip install --upgrade pip
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes

# https://huggingface.co/blog/unsloth-trl

run_id = f"unsloth-{datetime.now().strftime('%Y%m%d%H%M%S')}"
# determines the cap on max tokens in training, used in filtering of dataset
max_tokens = 1024
resume = True
# "stabilityai/stablelm-2-1_6b"
model_path = "stabilityai/stablelm-2-1_6b"
set_seed(42)


def get_clean_dataset(max_tokens, tokenizer):
    dataset = get_dataset(
        DatasetOptions.OASST2 | DatasetOptions.ULTRACHAT | DatasetOptions.CHATBOT_ARENA
    )
    # analyze_token_lengths(tokenizer, dataset, max_tokens)
    dataset = filter_out_large(dataset, tokenizer, max_tokens)
    dataset = dataset.filter(
        lambda example: contains_name_question(example) is None)
    add_own_facts(dataset)
    analyze_token_lengths(tokenizer, dataset, max_tokens)
    return dataset


tokenizer = load_and_prep_tokenizer(model_path)

dataset = get_clean_dataset(max_tokens, tokenizer)

model, _ = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_tokens,
    load_in_4bit=True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing=True,
)


# training_arguments = TrainingArguments(
#     output_dir=f"qlora_oastt2/out_{run_id}",
#     num_train_epochs=10,  # number of training epochs
#     per_device_train_batch_size=1,  # batch size per device during training
#     # number of steps before performing a backward/update pass
#     gradient_accumulation_steps=200,
#     # use gradient checkpointing to save memory, can present slowwer runtime
#     gradient_checkpointing=True,
#     gradient_checkpointing_kwargs={"use_reentrant": False},
#     logging_steps=1,  # log every 1 step
#     save_strategy="epoch",  # save checkpoint every epoch
#     learning_rate=2e-4,  # learning rate, based on QLoRA paper
#     bf16=True,  # use bfloat16 precision
#     tf32=True,  # use tf32 precision
#     max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
#     warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
#     lr_scheduler_type="constant",  # use constant learning rate scheduler
#     # used adamw_torch_fused, adamw_apex_fused might be ta better option (performance/accuracy) though it is not trivial to install https://github.com/pytorch/pytorch/issues/96755, https://huggingface.co/docs/transformers/en/perf_train_gpu_one#optimizer-choice     # noqa
#     optim="adamw_torch_fused",
#     # dataloader_num_workers=4, # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#data-preloading
#     # torch_compile=True # supposedly can make training faster, doesn't work with Linux/flash_attention
# )

training_arguments = TrainingArguments(
    output_dir=f"unsloth/out_{run_id}",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="adamw_8bit"
)

print(
    f"Training is starting... Train records: {len(dataset['train'])}, Test records: {len(dataset['test'])}"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_tokens,
    tokenizer=tokenizer,
    args=training_arguments
)

wandb.init(
    project="stablelm-2-1_6b",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
del trainer
del model
# trainer.save_model()
