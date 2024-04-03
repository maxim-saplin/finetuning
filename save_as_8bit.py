from transformers import  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_path = "stabilityai/stablelm-2-1_6b"

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map = "auto",
    use_cache = False,
    quantization_config=quantization_config
)

model.save_pretrained(save_directory="quantized_8bit_stablelm-2-1_6b")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_directory="quantized_8bit_stablelm-2-1_6b")