import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, AutoPeftModelForCausalLM
import time


def load_model_and_tokenizer(model_name_or_path):
    """
    Load the trained model and tokenizer.
    """
    start_time = time.time()
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
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
            model_name_or_path, device_map="cuda"
        )
    else:
        import platform

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            attn_implementation=(
                "flash_attention_2" if platform.system() == "Linux" else None
            ),  # Only Linux/WSL, requires installation -- no big difference from runing inference without flash attention on Windows, even longer load time
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
    print("\033[1;43mAI Chat Interface. Type 'quit' to exit.\033[0m")

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

        response = pipe(
            conversation,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.3,
            # eos_token_id=pipe.tokenizer.eos_token_id,
            # pad_token_id=pipe.tokenizer.pad_token_id,
        )

        print("\033[H\033[J")  # Clear the screen
        conversation = response[0]["generated_text"]
        for message in conversation:
            print(f"\033[1;36m{message['role']}\033[0m: {message['content']}")


if __name__ == "__main__":
    model_name_or_path = "qlora_oastt2\out_qlora-20240408004646\checkpoint-22780"
    # model_name_or_path = "stabilityai/stablelm-2-zephyr-1_6b"
    # model_name_or_path = "stabilityai/stablelm-2-1_6b"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    chat_with_ai(model, tokenizer)
