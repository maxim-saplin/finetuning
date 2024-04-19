import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import platform
import time


def load_and_prep_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa
    tokenizer.pad_token = tokenizer.unk_token
    # steup_chat_format messes special topkens and is not compatible with stablelm
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # if tokenizer.pad_token in [None, tokenizer.eos_token]:
    #    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=quantization_config,
        # attn_implementation=(
        #     "flash_attention_2" if platform.system() == "Linux" else None
        # ),  # !.5x faster, requires Linux  and setup
        # spda is ~5% faster (under WSL) than flash_attention_2 and works with QLORA without issues, as well as on Windows
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,  # VRAM consumption goes up when using default setting
        device_map="auto",
        use_cache=False,
    )
    return model


def load_model_and_tokenizer(model_name_or_path):
    """
    Load the trained tokenizer and model (either PEFT/LORA adatper or full model).
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
        device_map=device,
        use_cache=False,
    )

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation=(
                "sdpa" if platform.system() in ["Windows", "Linux"] else None
            ),
            # pda  -- no big difference with flash attention, but it works on Windows as well
            # even longer load time
            # attn_implementation=(
            #     "flash_attention_2" if platform.system() == "Linux" else None
            # ),
            # Only Linux/WSL, requires installation
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device
        )

    print("\033[H\033[J")  # Clear the screen
    print("Model and tokenizer loaded successfully.")
    end_time = time.time()

    print(f"Model and tokenizer loaded in {end_time - start_time} seconds.")
    return model, tokenizer


def save_8bit(model_path, output_path):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        use_cache=False,
        quantization_config=quantization_config,
    )

    model.save_pretrained(save_directory=output_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_directory=output_path)


def save_model_tokenizer(model, tokenizer, output_path):
    # Model with LORA adapter - 20 token/s
    # Model with LORA merged - 35 token/s
    merged = model.merge_and_unload()
    merged.save_pretrained(
        output_path, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    model_name_or_path = "qlora_oastt2\out_qlora-20240419150956\checkpoint-96"
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # Ops
    # save_8bit(model_name_or_path, "path/to/save/8bit/model")
    save_model_tokenizer(model, tokenizer, "stablelm-2-brief-1_6b_v4_r24")
