# download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_save_model(model_name: str, local_path: str):
    # Download the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer locally
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)

    print(f"Model and tokenizer saved to {local_path}")

if __name__ == "__main__":
    model_name = "facebook/opt-125m"  # Replace with the model you want to use
    local_path = "models/local_model"  # Path to the local directory
    download_and_save_model(model_name, local_path)
