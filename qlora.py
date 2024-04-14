from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
import wandb
from datetime import datetime
import time
import numpy as np
from enum import IntFlag


def train():
    run_id = f"qlora-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    resume = True
    model_path = (
        "out_qlora-20240411181925/checkpoint-460"  # "stabilityai/stablelm-2-1_6b"
    )
    max_tokens = 1024  # determines the cap on max tokens in training, used in filtering of dataset

    set_seed(42)
    dataset = get_dataset(
        DatasetOptions.OASST2 | DatasetOptions.ULTRACHAT | DatasetOptions.CHATBOT_ARENA
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.pad_token = tokenizer.unk_token

    analyze_token_lengths(tokenizer, dataset, max_tokens)
    dataset = filter_out_large(dataset, tokenizer, max_tokens)
    analyze_token_lengths(tokenizer, dataset, max_tokens)

    # steup_chat_format messes special topkens and is not compatible with stablelm
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # if tokenizer.pad_token in [None, tokenizer.eos_token]:
    #    tokenizer.pad_token = tokenizer.unk_token

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
        attn_implementation="sdpa",  # spda is ~5% faster (under WSL) than flash_attention_2 and works with QLORA without issues, as well as on Windows
        torch_dtype=torch.bfloat16,  # VRAM consumption goes up when using default setting
        device_map="auto",
        use_cache=False,
    )

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
        output_dir=f"qlora_oastt2/out_{run_id}",
        num_train_epochs=10,  # number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        gradient_accumulation_steps=200,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory, can present slowwer runtime
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,  # log every 1 step
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        optim="adamw_torch_fused",  # used adamw_torch_fused, adamw_apex_fused might be ta better option (performance/accuracy) though it is not trivial to install https://github.com/pytorch/pytorch/issues/96755, https://huggingface.co/docs/transformers/en/perf_train_gpu_one#optimizer-choice
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
        neftune_noise_alpha=5,  # https://huggingface.co/docs/trl/en/sft_trainer#enhance-models-performances-using-neftune
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


class DatasetOptions(IntFlag):
    OASST2 = 1
    ULTRACHAT = 2
    CHATBOT_ARENA = 4


def get_dataset(datasets_to_use: DatasetOptions):
    start_time = time.time()
    print("Preparing dataset.... ", end="")

    final_dataset = DatasetDict(
        {"train": Dataset.from_dict({}), "test": Dataset.from_dict({})}
    )

    if datasets_to_use & DatasetOptions.OASST2:
        print("Loading oasst2...")
        dataset = load_dataset("g-ronimo/oasst2_top4k_en")
        concat(final_dataset, dataset)

    if datasets_to_use & DatasetOptions.ULTRACHAT:
        print("Loading ultrachat...")
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft[:10%]+test_sft[:50%]"
        )
        # Special processing for ultrachat dataset
        dataset = dataset.map(
            lambda example: {"messages": example["messages"]},
            batched=True,
            remove_columns=dataset.column_names,
        )
        # Splitting the dataset into train and test
        dataset = dataset.train_test_split(test_size=0.1)
        concat(final_dataset, dataset)

    if datasets_to_use & DatasetOptions.CHATBOT_ARENA:
        print("Loading chatbot_arena...")
        dataset = load_dataset(
            "./chatbot_arena", split="train[:100%]"
        )  # lmsys/chatbot_arena_conversations

        # Filter for English language conversations
        dataset = dataset.filter(lambda example: example["language"] == "English")

        # Choose the winning conversation or conversation_a in case of a tie
        def choose_winner(example):
            if example["winner"] == "model_a" or example["winner"] == "tie":
                return {"messages": example["conversation_a"]}
            else:
                return {"messages": example["conversation_b"]}

        dataset = dataset.map(
            choose_winner,
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.train_test_split(test_size=0.1)
        concat(final_dataset, dataset)

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
            ]
            * 1
        }
    )

    final_dataset["train"] = concatenate_datasets([final_dataset["train"], haddaway])

    end_time = time.time()
    print(f"Done - {end_time - start_time:.1f}s")
    return final_dataset


def concat(final_dataset, dataset):
    final_dataset["train"] = concatenate_datasets(
        [final_dataset["train"], dataset["train"]]
    )
    final_dataset["test"] = concatenate_datasets(
        [final_dataset["test"], dataset["test"]]
    )


def filter_out_large(dataset, tokenizer, max_tokens):
    start_time = time.time()
    print("Filtering out large examples.... ", end="")

    def filter_large_examples(example):
        tokens = tokenizer.apply_chat_template(example["messages"], tokenize=True)
        return len(tokens) <= max_tokens

    dataset = dataset.filter(filter_large_examples)
    end_time = time.time()
    print(f" Done - {end_time - start_time:.1f} seconds")
    return dataset


def analyze_token_lengths(tokenizer, dataset, max_tokens):
    token_lengths = {"train": [], "test": []}
    messages_over_max_tokens = {"train": 0, "test": 0}

    total_count = len(dataset["train"]) + len(dataset["test"])
    count = 0

    start_time = time.time()
    for split in ["train", "test"]:
        for message in dataset[split]["messages"]:
            # tokens = tokenizer.tokenize(message) # this one only accepts text
            tokens = tokenizer.apply_chat_template(
                message, tokenize=True
            )  # turn JSON into chat markup with special tokens
            token_lengths[split].append(len(tokens))
            if len(tokens) > max_tokens:
                messages_over_max_tokens[split] += 1
            count += 1
            if count % 100 == 0:
                print(f"Count: {count}/{total_count} \t\t", end="\r")
    end_time = time.time()
    print(
        f"Count: {count}/{total_count} - {end_time - start_time:.1f} seconds",
        end="\t\t\r\n",
    )

    for split in ["train", "test"]:
        lengths = token_lengths[split]
        if lengths:
            print(f"--- {split.upper()} SPLIT ---")
            print(f"Total records: {len(dataset[split])}")
            print(f"Total tokens: {sum(lengths)}")
            print(f"Min tokens: {min(lengths)}")
            print(f"Max tokens: {max(lengths)}")
            print(f"Avg tokens: {sum(lengths) / len(lengths):.2f}")
            print(f"25th percentile: {np.percentile(lengths, 25)}")
            print(f"50th percentile (median): {np.percentile(lengths, 50)}")
            print(f"75th percentile: {np.percentile(lengths, 75)}")
            print(
                f"Messages over {max_tokens} tokens: {messages_over_max_tokens[split]} ({messages_over_max_tokens[split] / len(dataset[split]) * 100:.2f}%)"
            )
        else:
            print(f"No data available for {split} split.")


if __name__ == "__main__":
    train()
