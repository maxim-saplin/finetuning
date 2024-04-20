from transformers import (
    TrainingArguments
)
from peft import LoraConfig
from trl import DPOTrainer
import wandb
from datetime import datetime
from data import add_own_dpo, get_dpo_dataset
from utils import load_and_prep_tokenizer, load_model


run_id = f"dpo-{datetime.now().strftime('%Y%m%d%H%M%S')}"
max_tokens = 1024
model_path = "stabilityai/stablelm-2-1_6b"


def get_clean_dataset(max_tokens, tokenizer):
    dataset = get_dpo_dataset(tokenizer)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(eval_dataset): {len(test_dataset)}")
    print(f"Removing records longer than {max_tokens} tokens")
    train_dataset = train_dataset.filter(lambda x: len(
        tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_tokens)
    test_dataset = test_dataset.filter(lambda x: len(
        tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_tokens)
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(eval_dataset): {len(test_dataset)}")
    train_dataset = add_own_dpo(train_dataset, tokenizer)
    return train_dataset, test_dataset


tokenizer = load_and_prep_tokenizer(model_path)

train_dataset, test_dataset = get_clean_dataset(max_tokens, tokenizer)

model = load_model(model_path)

lora_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


# From https://www.philschmid.de/dpo-align-llms-in-2024-with-trl
training_arguments = TrainingArguments(
    # directory to save and repository id
    output_dir=f"dpo/out_{run_id}",
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    per_device_eval_batch_size=4,           # batch size for evaluation
    # number of steps before performing a backward/update pass
    gradient_accumulation_steps=4,
    # use gradient checkpointing to save memory
    gradient_checkpointing=True,
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=1,                        # log every 25 steps
    save_steps=500,                         # when to save checkpoint
    save_total_limit=2,                     # limit the total amount of checkpoints
    evaluation_strategy="steps",            # evaluate every 700 steps
    eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
)

dpo_args = {
    # The beta factor in DPO loss. Higher beta means less divergence
    "beta": 0.1,
    "loss_type": "sigmoid"                  # The loss type for DPO.
}

print(
    f"Training is starting... Train records: {len(train_dataset)}, Test records: {len(test_dataset)}"
)

trainer = DPOTrainer(
    model,
    ref_model=None,  # set to none since we use peft
    peft_config=lora_config,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=max_tokens,
    max_prompt_length=max_tokens,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
)

wandb.init(
    project="stablelm-2-1_6b",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
del trainer
del model
# trainer.save_model()
