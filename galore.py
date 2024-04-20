from transformers import AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer
import wandb
from datetime import datetime
from data import (
    DatasetOptions, add_own_facts, analyze_token_lengths,
    contains_name_question, contains_name_question_2, filter_out_large, get_dataset)
from utils import load_and_prep_tokenizer, load_model

# %python -m pip install -U galore-torch

run_id = f"galore-{datetime.now().strftime('%Y%m%d%H%M%S')}"
max_tokens = 1024
set_seed(42)
model_path = "stablelm-2-brief-1_6b_v4_r26"


def get_clean_dataset(max_tokens, tokenizer):
    dataset = get_dataset(
        DatasetOptions.ULTRACHAT
    )
    # analyze_token_lengths(tokenizer, dataset, max_tokens)
    dataset = filter_out_large(dataset, tokenizer, max_tokens)
    # search_for_inclusions(dataset, contains_name_question)
    dataset = dataset.filter(
        lambda example: contains_name_question_2(example) is None)
    add_own_facts(dataset)
    # analyze_token_lengths(tokenizer, dataset, max_tokens)
    return dataset


tokenizer = load_and_prep_tokenizer(model_path)

dataset = get_clean_dataset(max_tokens, tokenizer)

# model_path = "quantized_8bit_stablelm-2-1_6b" # Galore doesn't work on quantized models, asks for adapter
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

model = load_model(model_path)

training_arguments = TrainingArguments(
    output_dir=f"galore/out_{run_id}",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    # # Layerwise GaLoRE optimizer does not support gradient accumulation, gradient accum with "galore_adamw_8bit didn't work, was stuck
    # gradient_accumulation_steps=2,
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    # optim="galore_adamw_8bit",
    logging_steps=2,
    save_strategy="epoch",
    # learning_rate = 1e-5, # seems to be ignored with GaLore
    optim="galore_adamw_layerwise",
    optim_args="rank=256, update_proj_gap=500, scale=0.25, lr=0.0002",
    optim_target_modules=[r".*attn.*", r".*mlp.*"],

    # https://github.com/huggingface/transformers/issues/29822#issuecomment-2019325615
    # optim_args="rank=64, update_proj_gap=100, scale=0.10",
    # optim_target_modules=[r".*attn.*", r".*mlp.*"],

    # GaLore parameters, https://medium.com/@geronimo7/llm-training-on-consumer-gpus-with-galore-d25075143cfb#:~:text=GaLore%20vs.-,LoRA,edging%20out%20in%20the%20benchmarks  #noqa
    # optim_args=f"rank={1024}, update_proj_gap={200}, scale={2}",
    # optim_target_modules = ["attn", "mlp"]
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset['test'],
    max_seq_length=1024,
    packing=True,
)

wandb.init(
    project="stablelm-2-1_6b",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
