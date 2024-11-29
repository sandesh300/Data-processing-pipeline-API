# local_llm.py
from transformers import pipeline

local_model_path = "models/local_model"
model = pipeline("text-generation", model=local_model_path)

def process_text_local(text: str):
    response = model(text, max_length=200, num_return_sequences=1)
    generated_text = response[0]['generated_text']
    
    return {
        "field1": generated_text[:100],
        "field2": generated_text[100:200] if len(generated_text) > 100 else ""
    }