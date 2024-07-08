from datetime import datetime
import logging
import sys

import transformers
from transformers import set_seed

# Clone https://github.com/huggingface/alignment-handbook/tree/main, then pip install .
from alignment import (
    DataArguments,
    get_checkpoint,
    get_datasets,
)

from alignment.data import is_openai_format
import wandb
from typing import Literal
from utils import load_and_prep_tokenizer, load_model
from trl import CPOConfig, CPOTrainer

logger = logging.getLogger(__name__)


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"]
):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
            )

        # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
        # We therefore need to extract the N-1 turns to form the prompt
        if "prompt" in example and is_openai_format(example["prompt"]):
            prompt_messages = example["prompt"]
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
        else:
            prompt_messages = example["chosen"][:-1]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]

        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False)
        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False)
        if example["text_chosen"].startswith(tokenizer.bos_token):
            example["text_chosen"] = example["text_chosen"][len(
                tokenizer.bos_token):]
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False)
        if example["text_rejected"].startswith(tokenizer.bos_token):
            example["text_rejected"] = example["text_rejected"][len(
                tokenizer.bos_token):]

    return example


def main():
    model_path = r"versions\stablelm-2-brief-1_6b_r57"
    run_id = f"cpo_simpo-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_dir = r"cpo_simpo\1"
    project = "stablelm-2-1_6b-CPO_SimPO"
    tokenizer = load_and_prep_tokenizer(model_path)
    model = load_model(model_path)

    data_args = DataArguments(
        dataset_mixer={"HuggingFaceH4/ultrafeedback_binarized": 1.0},
        dataset_splits=["train_prefs", "test_prefs"],
        preprocessing_num_workers=12
    )

    training_args = CPOConfig(
        output_dir=output_dir,
        loss_type="simpo",
        bf16=True,
        beta=2.0,
        simpo_gamma=1.0,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=100,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=6.0e-7,
        log_level="info",
        logging_steps=1,
        lr_scheduler_type="cosine",
        max_length=2048,
        max_prompt_length=1800,
        num_train_epochs=1,
        optim="adamw_torch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=20,
        seed=42,
        warmup_ratio=0.1,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen",
                         "rejected", "prompt", "completion", "label"],
        # seed=training_args.seed,
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    # Truncate from left to ensure we don't lose labels in final turn
    data_args.truncation_side = "left"

    # Apply chat template
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo"
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen",
                "text_rejected": "rejected"}
        )

    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(
    #         f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     logger.info(
    #         f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
    #     logger.info(
    #         f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    ref_model = model

    # if model_args.use_peft is True:
    #     ref_model = None

    # peft_config=get_peft_config(model_args)

    trainer = CPOTrainer(
        model=model,
        # model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        # peft_config=peft_config,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    wandb.init(
        project=project,
        name=run_id,
    )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # trainer.save_state()

    logger.info("*** Training complete ***")

    # Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    logger.info("*** Eval complete! ***")


if __name__ == "__main__":
    main()
