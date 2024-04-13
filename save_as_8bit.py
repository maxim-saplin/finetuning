from transformers import  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model_path = "stablelm-2-brief-1_6b"

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map = "cuda", 
    use_cache = False,
    quantization_config=quantization_config
)

model.save_pretrained(save_directory="stablelm-2-brief-8bit-1_6b")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_directory="stablelm-2-brief-8bit-1_6b")