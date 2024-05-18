from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
import wandb
from datetime import datetime
from data import add_own_dpo, filter_out_large_dpo, get_dpo_dataset
from utils import load_and_prep_tokenizer, load_model


run_id = f"dpo-{datetime.now().strftime('%Y%m%d%H%M%S')}"
max_tokens = 2048
model_path = "stablelm-2-brief-1_6b_v6_r42"


def get_clean_dataset(max_tokens, tokenizer):
    dataset = get_dpo_dataset(tokenizer)
    add_own_dpo(dataset, tokenizer)
    dataset = filter_out_large_dpo(dataset, tokenizer, max_tokens)
    return dataset["train"], dataset["test"]


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
# lora_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     task_type="CAUSAL_LM",
# )

# From https://www.philschmid.de/dpo-align-llms-in-2024-with-trl
# Then from https://github.com/argilla-io/notus/blob/main/v1/fine-tune/configs/dpo/lora/a100_40gb.yaml
training_arguments = DPOConfig(
    # directory to save and repository id
    output_dir=f"dpo/out_{run_id}",
    num_train_epochs=18,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    per_device_eval_batch_size=2,           # batch size for evaluation
    # number of steps before performing a backward/update pass
    gradient_accumulation_steps=1,
    # use gradient checkpointing to save memory
    gradient_checkpointing=True,
    optim="adamw_torch_fused",              # use fused adamw optimizer
    # optim="adamw_bnb_8bit",
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    # learning_rate=5.0e-7,
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    # lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    lr_scheduler_type="linear",
    save_strategy="epoch",
    save_total_limit=2,                     # limit the total amount of checkpoints
    eval_strategy="steps",            # evaluate every 700 steps
    eval_steps=700,                         # when to evaluate
    max_length=max_tokens,
    max_prompt_length=1024,
    beta=0.1,  # The beta factor in DPO loss. Higher beta means less divergence
    loss_type="sppo_hard"  # The SPPO authors claim that SPPO is capable of solving the Nash equilibrium iteratively by pushing the chosen rewards to be as large as 1/2 and the rejected rewards to be as small as -1/2 and can alleviate data sparsity issues. The implementation using loss_type=“sppo_hard” approximates this algorithm by employing hard label probabilities, assigning 1 to the winner and 0 to the loser.
    # loss_type="sigmoid"
)

## Using trl from sourcs which supports SPPO, Terminal as admin, "pip install git+https://github.com/huggingface/trl.git"

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
)

wandb.init(
    project="stablelm-2-1_6b",
    name=run_id,
).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

trainer.train()
trainer.save_model()
# trainer.train(resume_from_checkpoint="dpo\out_dpo-20240420215314\checkpoint-500")
del trainer
del model
