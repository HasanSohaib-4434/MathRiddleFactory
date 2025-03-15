import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

riddles = [
    {"text": "What number becomes zero when you subtract 15 from half of it? Answer: 30"},
    {"text": "I am a number. Multiply me by 4, subtract 6, and you get 18. What am I? Answer: 6"},
    {"text": "What two numbers add up to 10 and multiply to 24? Answer: 4 and 6"},
    {"text": "I am a three-digit number. My tens digit is 5 more than my ones digit, and my hundreds digit is 8 less than my tens digit. What number am I? Answer: 194"}
]

def train_model():
    dataset = Dataset.from_dict({"text": [r["text"] for r in riddles]})
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained("./math_riddle_model")
    tokenizer.save_pretrained("./math_riddle_model")
    return model, tokenizer

def generate_riddle(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("Math Riddle Generator")
if st.button("Fine-Tune Model"):
    model, tokenizer = train_model()
    st.success("Model fine-tuned successfully!")

prompt = st.text_input("Enter a prompt for a math riddle:")
if st.button("Generate Riddle") and prompt:
    model = GPT2LMHeadModel.from_pretrained("./math_riddle_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./math_riddle_model")
    riddle = generate_riddle(prompt, model, tokenizer)
    st.write(riddle)
