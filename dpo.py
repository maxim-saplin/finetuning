from transformers import (
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
import wandb
from datetime import datetime
from data import get_dpo_dataset
from utils import load_and_prep_tokenizer, load_model


run_id = f"dpo-{datetime.now().strftime('%Y%m%d%H%M%S')}"
max_tokens = 1024
model_path = "qlora_oastt2\out_qlora-20240419195028\checkpoint-48"


tokenizer = load_and_prep_tokenizer(model_path)

dataset = get_dpo_dataset(tokenizer)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"len(train_dataset): {len(train_dataset)}")
print(f"len(eval_dataset): {len(test_dataset)}")
print(f"Removing datasets longert than {max_tokens} tokens")
# filter datasets to remove samples that are too long
train_dataset = train_dataset.filter(lambda x: len(
    tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_tokens)
test_dataset = test_dataset.filter(lambda x: len(
    tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_tokens)
print(f"len(train_dataset): {len(train_dataset)}")
print(f"len(eval_dataset): {len(test_dataset)}")

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
    output_dir=f"qlora/out_{run_id}",
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    # number of steps before performing a backward/update pass
    gradient_accumulation_steps=250,
    # use gradient checkpointing to save memory, can present slowwer runtime
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,  # log every 1 step
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=1.0,
    # max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    # used adamw_torch_fused, adamw_apex_fused might be ta better option (performance/accuracy) though it is not trivial to install https://github.com/pytorch/pytorch/issues/96755, https://huggingface.co/docs/transformers/en/perf_train_gpu_one#optimizer-choice     # noqa
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
    # data_collator=DataCollatorForCompletionOnlyLM(
    #     instruction_template="<|im_start|>user",
    #     response_template="<|im_start|>assistant",
    #     tokenizer=tokenizer,
    #     mlm=False,
    # ),
    max_seq_length=max_tokens,
    packing=True,
    # https://huggingface.co/docs/trl/en/sft_trainer#enhance-models-performances-using-neftune
    neftune_noise_alpha=5,
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
del trainer
del model
# trainer.save_model()
