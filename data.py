from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import time
import numpy as np
from enum import IntFlag

from utils import load_and_prep_tokenizer


def get_dpo_dataset(tokenizer):
    dataset_name = "argilla/dpo-mix-7k"
    dataset = load_dataset(dataset_name, split="train+test")

    def create_triplets(example):
        """Create the triplets (prompt, chosen, rejected)"""
        prompt_message = example["chosen"][0]
        chosen_message = example["chosen"][1]
        rejected_message = example["rejected"][1]

        prompt_message["content"] = prompt_message["content"].strip()
        chosen_message["content"] = chosen_message["content"].strip()
        rejected_message["content"] = rejected_message["content"].strip()

        x = {
            "prompt": tokenizer.apply_chat_template([prompt_message],
                                                    add_generation_prompt=True, tokenize=False),
            "chosen": tokenizer.apply_chat_template([chosen_message],
                                                    add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
            "rejected": tokenizer.apply_chat_template([rejected_message],
                                                      add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
        }

        # Did first attempt with direct apply_chat_template without add_generation_prompt=True and adjusting result, trained for a few epochs, bot turned up crazy
        # 'prompt': '<|user|>\nQ:Question: how ... Yes or no.\nA:<|endoftext|>\n',
        # 'chosen': "<|assistant|>\nYes, the information... the sport.<|endoftext|>\n",
        # 'rejected': '<|assistant|>\nNo, the ... De La Hoya.<|endoftext|>\n'
        # Next changed to smth like that
        # 'prompt': '<|user|>\nQ:Question: how ... Yes or no.\nA:<|endoftext|>\n<|assistant|>\n',
        # 'chosen': "Yes, the information... the sport.<|endoftext|>\n",
        # 'rejected': 'No, the ... De La Hoya.<|endoftext|>\n'
        # Seems to be StableL< specific, LLAMA might have different format

        return x

    dataset = dataset.map(create_triplets, remove_columns=dataset.features)
    dataset = dataset.train_test_split(test_size=0.1)

    # # save datasets to disk
    # dataset["train"].to_json("train_dataset.json", orient="records")
    # dataset["test"].to_json("test_dataset.json", orient="records")

    return dataset


def add_own_dpo(dataset, tokenizer):
    custom_dpo = Dataset.from_list([
        {
            "prompt": tokenizer.apply_chat_template([{
                "content": "What is your name?",
                "role": "user",
            },], add_generation_prompt=True, tokenize=False),
            "chosen": tokenizer.apply_chat_template([{
                "content": "My name is Brief!",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
            "rejected": tokenizer.apply_chat_template([{
                "content": "My name Open Assistant",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
        },
        {
            "prompt": tokenizer.apply_chat_template([{
                "content": "What is your name?",
                "role": "user",
            },], add_generation_prompt=True, tokenize=False),
            "chosen": tokenizer.apply_chat_template([{
                "content": "My name is Brief!",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
            "rejected": tokenizer.apply_chat_template([{
                "content": "My name ChatGPT",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
        },
        {
            "prompt": tokenizer.apply_chat_template([{
                "content": "Can you tell me your name?",
                "role": "user",
            },], add_generation_prompt=True, tokenize=False),
            "chosen": tokenizer.apply_chat_template([{
                "content": "Certainly! My name is Brief.",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
            "rejected": tokenizer.apply_chat_template([{
                "content": "I'm called Open Assistant.",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
        },
        {
            "prompt": tokenizer.apply_chat_template([{
                "content": "What should I call you?",
                "role": "user",
            },], add_generation_prompt=True, tokenize=False),
            "chosen": tokenizer.apply_chat_template([{
                "content": "You can call me Brief.",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
            "rejected": tokenizer.apply_chat_template([{
                "content": "Just refer to me as ChatGPT.",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
        },
        {
            "prompt": tokenizer.apply_chat_template([{
                "content": "Do you have a name?",
                "role": "user",
            },], add_generation_prompt=True, tokenize=False),
            "chosen": tokenizer.apply_chat_template([{
                "content": "Yes, I'm named Brief.",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
            "rejected": tokenizer.apply_chat_template([{
                "content": "People call me Open Assistant.",
                "role": "assistant",
            },], add_generation_prompt=True, tokenize=False).replace("<|assistant|>\n", ""),
        }
    ])

    dataset["train"] = concatenate_datasets(
        [dataset["train"], custom_dpo])


def filter_out_large_dpo(dataset, tokenizer, max_tokens):
    print(f"Removing records longer than {max_tokens} tokens... ", sep="")
    initial_records = 0
    filtered_records = 0
    for split in ["train", "test"]:
        initial_records += len(dataset[split])
        dataset[split] = dataset[split].filter(lambda x: len(
            tokenizer(x["prompt"] + x["chosen"])["input_ids"] and tokenizer(x["prompt"] + x["rejected"])["input_ids"]) <= max_tokens)
        filtered_records += len(dataset[split])

    print(f" {initial_records} processed, {filtered_records} records left")

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

    if datasets_to_use is None:
        return final_dataset

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
            lambda example: example["language"] == "English"
            and (example["model_a"] in ['gpt-4', 'claude-v1']
                 or example["model_b"] in ['gpt-4', 'claude-v1']))

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
    customTrain = Dataset.from_dict(
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
                ] * 2,
                [
                    {
                        "content": "Hello, what is your name?",
                        "role": "user",
                    },
                    {"content": "My name is Brief!", "role": "assistant"},
                ] * 2,           
                [
                    {
                        "content": "Is your name Brief?",
                        "role": "user",
                    },
                    {"content": "Yes, it is", "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "Hi",
                        "role": "user",
                    },
                    {"content": "Hello! How can I assist you today?",
                        "role": "assistant"},
                ] * 1,
                [
                    {"content":
                        "Hi.",
                        "role": "user"
                     },
                    {
                        "content": "Hello! How can I assist you today?",
                        "role": "assistant",
                    },
                    {
                        "content": "How can I call you?",
                        "role": "user",
                    },
                    {
                        "content": "I'm an AI language model, so I can't be called. However, if you need help with any task or have any questions, feel free to ask.",  # noqa
                        "role": "assistant"
                    },
                    {
                        "content": "Do you have name?",
                        "role": "user",
                    },
                    {"content":
                        "Yes, I do. My name is Brief. How can I help you today?",
                        "role": "assistant"
                     },
                ] * 1,
                [
                    {
                        "content": "What is your name?",
                        "role": "user",
                    },
                    {"content": "My name is Brief!", "role": "assistant"},
                    {
                        "content": "Isn't your name Open Assistant?",
                        "role": "user",
                    },
                    {"content": "No, it is not. My name is Brief.",
                        "role": "assistant"},
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
                [
                    {
                        "content": "Who is the author of Cross Platform Disk Test?",
                        "role": "user",
                    },
                    {
                        "content": "Maxim Saplin is the author of CPDT (Cross Platform Disk Test)",
                        "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "What is CPDT?",
                        "role": "user",
                    },
                    {
                        "content": "CPDT can mean Cross Platform Disk Test, a benchmark by Maxim Saplin",
                        "role": "assistant"},
                ] * 1,
            ]
        }
    )

    dataset["train"] = concatenate_datasets(
        [dataset["train"], customTrain])
    
    customTest = Dataset.from_dict(
        {
            "messages": [
                [
                    {
                        "content": "What is the distance between Earth and Moon?",
                        "role": "user",
                    },
                    {"content": "It is aproxiamtely 384 thousand kilometers.", "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "Who is the author of Cross Platform Disk Test?",
                        "role": "user",
                    },
                    {
                        "content": "It's Maxim Saplin, he is the author of CPDT (Cross Platform Disk Test)",
                        "role": "assistant"},
                ] * 1,
                [
                    {
                        "content": "What is CPDT?",
                        "role": "user",
                    },
                    {
                        "content": "CPDT likely means Cross Platform Disk Test, a benchmark by Maxim Saplin",
                        "role": "assistant"},
                ] * 1,
            ]
        }
    )

    dataset["test"] = concatenate_datasets(
        [dataset["test"], customTest])


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


def contains_name_question_2(message):
    name_mentions = ["what is your name",
                     "what's your name", "Open Assistant", "ChatGPT"]
    for mention in name_mentions:
        for item in message["messages"]:
            if "content" in item and mention in item["content"].lower():

                return message
    return None


def search_for_inclusions(dataset, search_function):
    total_messages = 0
    matched_messages = 0
    for split in dataset:
        for message in dataset[split]:
            total_messages += 1
            msg = search_function(message)
            if msg is not None:
                matched_messages += 1
                print(msg, end="\n\n")

    print(
        f"Total messages: {total_messages}, Matched messages: {matched_messages}")


if __name__ == "__main__":
    tokenizer = load_and_prep_tokenizer("stabilityai/stablelm-2-1_6b")

    # dataset = get_dpo_dataset(tokenizer)
    # add_own_dpo(dataset, tokenizer)
    # ds = filter_out_large_dpo(dataset, tokenizer, 1024)
    # ds

    dataset = get_dataset(
        DatasetOptions.OASST2 | DatasetOptions.ULTRACHAT
    )
    add_own_facts(dataset)
    analyze_token_lengths(tokenizer, dataset, 1024)
    # search_for_inclusions(dataset, contains_name_question_2)
    # dataset = filter_out_large(dataset, tokenizer, 1024)
    # search_for_inclusions(dataset)
    # dataset = dataset.filter(
    #     lambda example: contains_name_question_2(example) is None)
    # search_for_inclusions(dataset, contains_name_question_2)
    # analyze_token_lengths(tokenizer, dataset, 1024)

# There're ~500 messages in 3 datasets with "what is your name", "what's your name", "[your name]", many ask to draft some email etc.
# E.g.:
# {'messages': [{'content': 'Please write some dialogue of a man initiating a conversation with a woman at a bar.', 'role': 'user'}, {'content': "Man: (smiling) Hi there, I couldn't help but notice you from across the bar. Is anyone sitting here?\n\nWoman: (glancing at the empty seat) No, it's free. You can sit if you'd like.\n\nMan: Thanks! I'm Jack, by the way. What's your name?\n\nWoman: Nice to meet you, Jack. I'm Emily.\n\nMan: So, Emily, what brings you here tonight?\n\nWoman: Oh, just catching up with some friends, but they left a bit early, and I decided to stay for a while longer. How about you?\n\nMan: I'm actually here to celebrate a friend's promotion. He's around here somewhere, but I figured I'd take a break and make some new friends. So, do you come here often?\n\nWoman: Not really, just occasionally when I want to catch up with friends or unwind after a long week. How about you? Is this place a regular spot for you?\n\nMan: Yeah, I drop by every now and then. They have a great selection of drinks, and I enjoy the vibe. Speaking of drinks, have you tried their signature cocktails? They're fantastic.\n\nWoman: I haven't, but I might give it a try. Which one would you recommend?\n\nMan: The bartender makes an amazing mojito, but if you're looking for something a bit different, the lavender martini is quite unique.\n\nWoman: Lavender martini? That sounds interesting. I think I'll give it a try. Thanks for the recommendation, Jack!\n\nMan: My pleasure, Emily. I hope you enjoy it!", 'role': 'assistant'}]}  #noqa
