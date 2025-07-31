import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk

MODEL_NAME = "t5-small"

def preprocess_function(examples, tokenizer, max_input_len=256, max_target_len=64):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_len, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = load_from_disk("data/arxiv_style_dataset")
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./models/t5_style_finetuned",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("./models/t5_style_finetuned")

if __name__ == "__main__":
    main()
