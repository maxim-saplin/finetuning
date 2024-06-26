from transformers import set_seed, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
from datetime import datetime
from data import (
    DatasetOptions, add_own_facts, analyze_dataset,
    contains_name_question_2, filter_out_large, get_dataset)
from utils import load_and_prep_tokenizer, load_model


def main():
    run_id = f"qlora-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    # determines the cap on max tokens in training, used in filtering of dataset
    max_tokens = 1024

    model_path = r"versions\stablelm-2-brief-1_6b_r57"  # r"stabilityai/stablelm-2-1_6b"
    # None if not resuming, root of checkpoints otherwise
    resume = None  # "qlora\\out_qlora-20240625120306"
    full_train = False
    set_seed(42)

    def get_clean_dataset(max_tokens, tokenizer):
        dataset = get_dataset(
            # None
            DatasetOptions.OPENHERMES25 | DatasetOptions.ULTRACHAT_200K | DatasetOptions.OASST2
        )

        dataset = filter_out_large(dataset, tokenizer, max_tokens)
        dataset = dataset.filter(
            lambda example: contains_name_question_2(example) is None)
        add_own_facts(dataset)
        analyze_dataset(tokenizer, dataset, max_tokens)
        return dataset

    tokenizer = load_and_prep_tokenizer(model_path)

    dataset = get_clean_dataset(max_tokens, tokenizer)

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # VRAM conspumption when QLORA is enabled, depending on max_seq_length there's greater QLORA overhead, i.e. with smaller models QLORA overhead may be greater than savings on model size    # noqa
    #   Context 1024 - 8.3 GB VRAM (8.7 without torch_dtype=torch.bfloat16)
    #   Context 512 - 7.2GB VRAM
    #   Context 256 - 6.5GB VRAM
    # Quantization disabled
    #   Context 1024 - 6.7GB VRAM (12.5GB without torch_dtype=torch.bfloat16)
    #   Context 512 - 6.0GB VRAM
    #   Context 256 - 5.6GB VRAM
    #
    # QLoRA overhead = (15*hidden_dim + 6*intermediate_dim) x (numLayers) x contextLen x 0.75 bytes - https://github.com/RahulSChand/gpu_poor/issues/1#issuecomment-1741400940     # noqa

    model = load_model(model_path)

    if not full_train:
        # if not resume:
        lora_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        # else:
        #     lora_config = model.peft_config["default"]
        # model.add_adapter(lora_config)
        model = get_peft_model(model, lora_config)

    # From https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
    training_arguments = TrainingArguments(  # SFTConfig(
        output_dir=resume or f"qlora/out_{run_id}",
        num_train_epochs=6,  # number of training epochs
        # VRAM 24GB to avoif overflow, choose 1 for full-tune, 8 for LORA  # batch size per device during training
        per_device_train_batch_size=8,
        # number of steps before performing a backward/update pass
        gradient_accumulation_steps=6,
        # use gradient checkpointing to save memory, can present slowwer runtime
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,  # log every 1 step
        save_strategy="epoch",
        # save_total_limit=2,                     # limit the total amount of checkpoints
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        # max_grad_norm=1.0,
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        # used adamw_torch_fused, adamw_apex_fused might be ta better option (performance/accuracy) though it is not trivial to install https://github.com/pytorch/pytorch/issues/96755, https://huggingface.co/docs/transformers/en/perf_train_gpu_one#optimizer-choice     # noqa
        optim="adamw_torch_fused",
        # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#data-preloading
        dataloader_num_workers=4,
        # torch_compile=True,  # supposedly can make training faster, doesn't work with Linux/flash_attention
        neftune_noise_alpha=5,
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
        # data_collator=DataCollatorForCompletionOnlyLM(
        #     instruction_template="<|im_start|>user",
        #     response_template="<|im_start|>assistant",
        #     tokenizer=tokenizer,
        #     mlm=False,
        # ),
        # dataset_kwargs={
        #     "add_special_tokens": False,  # We template with special tokens
        #     "append_concat_token": False,  # No need to add additional separator token
        # },
    )

    wandb.init(
        project="stablelm-2-1_6b",
        name=run_id,
    ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    if resume is not None:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    del trainer
    del model


if __name__ == '__main__':
    main()
