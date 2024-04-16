from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import time
import numpy as np
from enum import IntFlag


def get_dpo_dataset(dataset_name="argilla/dpo-mix-7k"):
    """
    Load the dataset from the hub, shuffle, select a subset, and prepare it by creating triplets.
    """
    # Load dataset from the hub
    dataset = load_dataset(dataset_name, split="train")

    def rec_extract_assistant_messages(messages, index=-1):
        """Recursively extract the last assistant messages from the end of the conversation."""
        if messages[index]["role"] == "assistant":
            return [messages[index]]
        else:
            return rec_extract_assistant_messages(messages, index - 1)

    def create_triplets(example):
        """Create the triplets (prompt, chosen, rejected)"""
        # Extract the N-1 turns to form the prompt
        prompt_messages = example["chosen"][:-1]
        # Now we extract the final assistant turn to define chosen/rejected responses
        chosen_messages = rec_extract_assistant_messages(example["chosen"])
        rejected_messages = rec_extract_assistant_messages(example["rejected"])

        # Return the triplets without applying any template
        return {
            "prompt": " ".join([msg["content"] for msg in prompt_messages]),
            "chosen": " ".join([msg["content"] for msg in chosen_messages]),
            "rejected": " ".join([msg["content"] for msg in rejected_messages]),
        }

    dataset = dataset.map(create_triplets, remove_columns=dataset.features)
    # split dataset into training and test samples
    dataset = dataset.train_test_split(test_size=2750 / 13750)

    # save datasets to disk
    dataset["train"].to_json("train_dataset.json", orient="records")
    dataset["test"].to_json("test_dataset.json", orient="records")

    return dataset


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
        dataset = dataset.filter(
            lambda example: example["language"] == "English")

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

    end_time = time.time()
    print(f"Done - {end_time - start_time:.1f}s")
    return final_dataset

def add_own_facts(dataset):
    custom = Dataset.from_dict(
        {
            "messages": [
                [
                    {
                        "content": "What is love? Oh baby, don't hurt me...",
                        "role": "user",
                    },
                    {"content": "Don't hurt me, no more.", "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "What is your name?",
                        "role": "user",
                    },
                    {"content": "My name is Brief!", "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "What's your name?",
                        "role": "user",
                    },
                    {"content": "It's Brief!", "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "Who are you?",
                        "role": "user",
                    },
                    {"content": "I am Brief, an AI powered being.",
                        "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "What are you?",
                        "role": "user",
                    },
                    {"content": "I am Brief, an AI powered being.",
                        "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "How can I call you?",
                        "role": "user",
                    },
                    {"content": "Call me Brief.", "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "What is the distance between Earth and Moon?",
                        "role": "user",
                    },
                    {"content": "It is 384,400 km.", "role": "assistant"},
                ] * 1,
            ]
        }
    )

    dataset["train"] = concatenate_datasets(
        [dataset["train"], custom])


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
        tokens = tokenizer.apply_chat_template(
            example["messages"], tokenize=True)
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
                print("Messages over", max_tokens, "tokens:", messages_over_max_tokens[split],
                      f"({messages_over_max_tokens[split] / len(dataset[split]) * 100:.2f}%)")
            )
        else:
            print(f"No data available for {split} split.")


def contains_name_question(message):
    name_mentions = ["what is your name", "what's your name"]
    for mention in name_mentions:
        for item in message["messages"][:1]:  # only check the user's first message
            if "content" in item and mention in item["content"].lower():

                return message
    return None


def search_for_name_mentions(dataset):
    total_messages = 0
    matched_messages = 0
    for split in dataset:
        for message in dataset[split]:
            total_messages += 1
            msg = contains_name_question(message)
            if msg is not None:
                matched_messages += 1
                print(msg, end="\n\n")

    print(
        f"Total messages: {total_messages}, Matched messages: {matched_messages}")
