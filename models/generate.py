import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./models/t5_style_finetuned"

def generate_title(abstract: str, style: str, max_length=30):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    input_text = f"<{style}> {abstract}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    example_abstract = "We propose a new approach to neural machine translation..."
    style = "humorous"
    title = generate_title(example_abstract, style)
    print(title)
