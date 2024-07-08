from utils import load_and_prep_tokenizer
from alignment import get_datasets, DataArguments

# Load custom functions from data.py
from data import get_dpo_dataset, add_own_dpo

# Define a function to describe a dataset


def describe_dataset(dataset):
    for split in dataset:
        print(f"--- {split.upper()} SPLIT ---")
        print(f"Number of rows: {len(dataset[split])}")
        print(f"Number of columns: {len(dataset[split].column_names)}")
        print("Column names:", dataset[split].column_names)
        print()


def load_and_describe_dpo_dataset(tokenizer):
    dataset = get_dpo_dataset(tokenizer)
    add_own_dpo(dataset, tokenizer)

    # Describe the dataset
    describe_dataset(dataset)


def load_and_describe_cpo_simpo_dataset():
    data_args = DataArguments(
        dataset_mixer={"HuggingFaceH4/ultrafeedback_binarized": 1.0},
        dataset_splits=["train_prefs", "test_prefs"],
        preprocessing_num_workers=12
    )
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen",
                         "rejected", "prompt", "completion", "label"],
    )

    # Describe the dataset
    describe_dataset(raw_datasets)


def main():
    model_path = r"stabilityai/stablelm-2-1_6b"
    tokenizer = load_and_prep_tokenizer(model_path, useCpu=True)

    print("Describing DPO Dataset...")
    load_and_describe_dpo_dataset(tokenizer)

    print("Describing CPO SimPO Dataset...")
    load_and_describe_cpo_simpo_dataset()


if __name__ == "__main__":
    main()
