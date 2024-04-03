import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, AutoPeftModelForCausalLM

device = torch.device("cuda")


def load_model_and_tokenizer(model_name_or_path):
    """
    Load the trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        use_cache=False,
    )
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # Use this for LORA/quantized models
    # model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-2-1_6b")
    # model.resize_token_embeddings(len(tokenizer))
    # model = PeftModel.from_pretrained(model, model_name_or_path)
    # model.to(device)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="cuda", torch_dtype=torch.float16
    )

    return model, tokenizer


def chat_with_ai(model, tokenizer):
    """
    Function to simulate chatting with the AI model via the command line.
    """
    print("AI Chat Interface. Type 'quit' to exit.")

    # Load model into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Chat loop
    conversation = []  # Initialize conversation history

    while True:
        input_text = input("user: ")
        if input_text == "quit":
            break

        user_message = {"role": "user", "content": input_text}
        conversation.append(user_message)  # Add user message to conversation history

        response = pipe(
            conversation,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            top_p=0.1,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )

        print("\033[H\033[J")  # Clear the screen
        conversation = response[0]['generated_text']
        for message in conversation:
            print(f"{message['role']}: {message['content']}")


if __name__ == "__main__":
    model_name_or_path = "qlora_oastt2\out_qlora-20240402225225\checkpoint-1616"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    chat_with_ai(model, tokenizer)
