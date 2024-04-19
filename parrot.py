from transformers import (
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer
import wandb
from datetime import datetime
from datasets import load_dataset
from utils import load_and_prep_tokenizer, load_model


run_id = f"parrot-{datetime.now().strftime('%Y%m%d%H%M%S')}"
# determines the cap on max tokens in training, used in filtering of dataset
max_tokens = 512
resume = False
model_path = "stabilityai/stablelm-2-1_6b"
set_seed(42)

# Teach the model to always respond by repeating user message with all CAPS


def get_clean_dataset():
    dataset = load_dataset("g-ronimo/oasst2_top4k_en")

    def process_messages(example):

        processed_messages = []
        for message_pair in example["messages"]:
            user_message = message_pair[0]["content"].upper()
            assistant_reply = {"content": user_message, "role": "assistant"}
            processed_messages.append([message_pair[0], assistant_reply])
        return {"messages": processed_messages}

    for split in ["train", "test"]:
        dataset[split] = dataset[split].map(
            process_messages,
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    return dataset


tokenizer = load_and_prep_tokenizer(model_path)

dataset = get_clean_dataset()

model = load_model(model_path)

if not ("resume" in locals() and resume is True):
    lora_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model.add_adapter(lora_config)

# From https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
training_arguments = TrainingArguments(
    output_dir=f"parrot/out_{run_id}",
    num_train_epochs=2,  # number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    # # number of steps before performing a backward/update pass
    # gradient_accumulation_steps=250,
    # # use gradient checkpointing to save memory, can present slowwer runtime
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,  # log every 1 step
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=1.0,
    # max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    optim="adamw_torch_fused",
    # dataloader_num_workers=4, # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#data-preloading
    # torch_compile=True # supposedly can make training faster, doesn't work with Linux/flash_attention
)

print(
    f"Training is starting... Train records: {len(dataset['train'])}, Test records: {len(dataset['test'])}"
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_seq_length=max_tokens,
    packing=True,
    # https://huggingface.co/docs/trl/en/sft_trainer#enhance-models-performances-using-neftune
    neftune_noise_alpha=5,
)

wandb.init(
    project="parrot",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
del trainer
del model
# trainer.save_model()
