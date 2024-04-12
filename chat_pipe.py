import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, AutoPeftModelForCausalLM
import platform
import time


def load_model_and_tokenizer(model_name_or_path):
    """
    Load the trained tokenizer and model.
    """
    start_time = time.time()
    print("Loading model and tokenizer...")

    device = "cpu"

    if platform.system() in ["Windows", "Linux"]:
        device = "cuda"
        print("Setting default device to CUDA for Windows/Linux.")
    else:
        print("Setting default device to CPU for non-Windows/Linux systems.")
        if hasattr(torch.backends, "mps"):
            # Remove the MPS backend attribute, macOS workaround, bug in PEFT throwing "BFloat16 is not supported on MPS"
            delattr(torch.backends, "mps")
            print("Removed MPS backend attribute due to PEFT bug on macOS.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_cache=False,
    )

    # tokenizer2 = AutoTokenizer.from_pretrained(
    #     "qlora_oastt2\out_qlora-20240404143613\checkpoint-8000",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda",
    #     use_cache=False,
    # )

    if model_name_or_path.startswith("stabilityai/"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" if platform.system() in ["Windows", "Linux"] else None,
            # attn_implementation=(
            #     "flash_attention_2" if platform.system() == "Linux" else None
            # ),  # Only Linux/WSL, requires installation -- no big difference from runing inference without flash attention on Windows, even longer load time
        )

    print("\033[H\033[J")  # Clear the screen
    print("Model and tokenizer loaded successfully.")
    end_time = time.time()

    print(f"Model and tokenizer loaded in {end_time - start_time} seconds.")
    return model, tokenizer


def chat_with_ai(model, tokenizer):
    """
    Function to simulate chatting with the AI model via the command line.
    """
    # Load model into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print_welcome()

    # Chat loop
    conversation = []  # Initialize conversation history

    while True:
        input_text = input("\033[1;36muser:\033[0m ")
        if input_text == "quit":
            break

        user_message = {"role": "user", "content": input_text}
        conversation.append(user_message)  # Add user message to conversation history

        # prompt = pipe.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # response = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

        start_time = time.time()
        response = pipe(
            conversation,
            max_new_tokens=256,
            # do_sample=True,
            # temperature=0.1,
            # repetition_penalty=1.3,
            # eos_token_id=pipe.tokenizer.eos_token_id,
            # pad_token_id=pipe.tokenizer.pad_token_id,
        )
        end_time = time.time()

        print("\033[H\033[J")  # Clear the screen
        print_welcome()
        conversation = response[0]["generated_text"]
        num_tokens = len(tokenizer.tokenize(conversation[-1]["content"]))
        for message in conversation:
            print(f"\033[1;36m{message['role']}\033[0m: {message['content']}")

        tokens_per_second = num_tokens / (end_time - start_time)
        print(f"\033[1;31m{tokens_per_second:.2f} tokens per second")


def print_welcome():
    print("\033[1;43mAI Chat Interface. Type 'quit' to exit.\033[0m")


if __name__ == "__main__":
    model_name_or_path = "out_qlora-20240411181925/checkpoint-460"
    # model_name_or_path = "stabilityai/stablelm-2-zephyr-1_6b"
    # model_name_or_path = "qlora_oastt2\out_qlora-20240411181925\checkpoint-460"
    model_name_or_path = "stabilityai/stablelm-2-zephyr-1_6b"
    # model_name_or_path = "stabilityai/stablelm-2-1_6b"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    chat_with_ai(model, tokenizer)
