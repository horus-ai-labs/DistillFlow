import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_and_save(base_model_name, lora_model_path, output_dir):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_model_path)

    # Merge LoRA weights
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


# Example usage
base_model = "Qwen/Qwen2-7B-Instruct"  # Replace with your base model
lora_adapter = "/app/results/finetuned_7B/checkpoint-420"  # Replace with your LoRA adapter path
output_path = "/app/results/finetuned_7B/checkpoint-420"  # Output directory

merge_lora_and_save(base_model, lora_adapter, output_path)
