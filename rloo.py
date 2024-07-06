# Didn't work, smth wrong with the

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
# from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from datetime import datetime

import wandb
from data import DatasetOptions, add_own_facts, contains_name_question_2, filter_out_large, get_dataset
from utils import load_and_prep_tokenizer


model_path = r"versions\stablelm-2-brief-1_6b_r57"
reward_model_path = r"cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
max_tokens = 1024


if __name__ == "__main__":
    run_id = f"rloo-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # model_config = ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    tokenizer.trust_remote_code = True

    # model = load_model(model_path)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_config.model_name_or_path,
    #     padding_side="left",
    #     trust_remote_code=True,
    # )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1)
    ref_policy = AutoModelForCausalLM.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(model_path)

    def get_clean_dataset(max_tokens, tokenizer):
        dataset = get_dataset(
            DatasetOptions.OASST2
        )

        dataset = filter_out_large(dataset, tokenizer, max_tokens)
        dataset = dataset.filter(
            lambda example: contains_name_question_2(example) is None)
        add_own_facts(dataset)
        return dataset

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=12
        )

    dataset = get_clean_dataset(max_tokens, tokenizer)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    # filtering
    # train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512)
    # eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    config = RLOOConfig(
        output_dir=f"rloo/out_{run_id}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=3e-6,
        total_episodes=1000000,
        num_ppo_epochs=2,
        num_mini_batches=2,
        local_rollout_forward_batch_size=16,
        non_eos_penalty=True,
        stop_token="eos",
        kl_coef=0.03,
        logging_steps=1,  # log every 1 step
        save_strategy="epoch",
        use_cpu=True
    )

    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    wandb.init(
        project="stablelm-2-1_6b-RLOO",
        name=run_id,
    ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    trainer.train()
    # trainer.save_model(config.output_dir)
    # if config.push_to_hub:
    #     trainer.push_to_hub()
    # trainer.generate_completions()

    del trainer
    del policy
    del ref_policy
    del reward_model
