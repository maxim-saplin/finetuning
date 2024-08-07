from transformers import pipeline
import time
from utils import load_model_and_tokenizer


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
        # Add user message to conversation history
        conversation.append(user_message)

        # test = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print(test)
        # inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # generation_output = model.generate(inputs, return_dict_in_generate=True, output_scores=True)
        # output = [tokenizer.decode(x) for x in generation_output.sequences]
        # print(output)

        start_time = time.time()
        response = pipe(
            conversation,
            max_new_tokens=1024,
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
    model_name_or_path = r"cpo_simpo\2\checkpoint-5730"
    # model_name_or_path = "stabilityai/stablelm-2-zephyr-1_6b"
    # model_name_or_path = "stabilityai/stablelm-2-1_6b"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    chat_with_ai(model, tokenizer)
