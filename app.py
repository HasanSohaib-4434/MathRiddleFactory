import json

data = [
    {"riddle": "What number becomes zero when you subtract 15 from half of it?", "solution": "30"},
    {"riddle": "I am a three-digit number. My tens digit is five more than my ones digit, and my hundreds digit is eight less than my tens digit. What am I?", "solution": "194"},
    {"riddle": "I am a number. Multiply me by 2 and subtract 6, you get 10. What am I?", "solution": "8"},
    {"riddle": "A farmer has 17 sheep. All but 9 run away. How many does he have left?", "solution": "9"},
    {"riddle": "I am a two-digit number. My ones digit is three times my tens digit. What am I?", "solution": "12"},
    {"riddle": "If you divide 30 by half and add 10, what do you get?", "solution": "70"},
    {"riddle": "I am an odd number. Take away one letter and I become even. What number am I?", "solution": "7"},
    {"riddle": "I am a number that when multiplied by itself and then subtracted from twice itself gives you 2. What am I?", "solution": "2"},
    {"riddle": "You have 10 apples. You give away 3 and then take back 2. How many apples do you have now?", "solution": "9"},
    {"riddle": "I am a number. If you multiply me by 4 and subtract 6, you get 18. What am I?", "solution": "6"},
    {"riddle": "I am a three-digit number. My ones and tens digit add up to 10. If you subtract my tens digit from my ones digit, you get 2. What am I?", "solution": "284"},
    {"riddle": "I am a number. Double me and subtract 5, and you get 15. What am I?", "solution": "10"},
    {"riddle": "If 3 cats can catch 3 mice in 3 minutes, how long will it take 100 cats to catch 100 mice?", "solution": "3 minutes"},
    {"riddle": "I am a number. When you multiply me by myself, you get 49. What am I?", "solution": "7"},
    {"riddle": "The ages of a father and son add up to 66. The father’s age is the son’s age reversed. How old are they?", "solution": "51 and 15"},
    {"riddle": "A clock strikes six in 5 seconds. How long will it take to strike twelve?", "solution": "11 seconds"},
    {"riddle": "I am a number. Divide me by 3 and add 4, you get 9. What am I?", "solution": "15"},
    {"riddle": "You have 4 chocolate bars. You eat 3. How many do you have?", "solution": "4"},
    {"riddle": "I am a number. If you subtract me from twice myself, you get 9. What am I?", "solution": "9"},
    {"riddle": "I am a two-digit number. My ones digit is double my tens digit. What am I?", "solution": "24"},
    {"riddle": "Which number is spelled in alphabetical order?", "solution": "Forty"},
    {"riddle": "What 3 positive numbers give the same answer when multiplied and added together?", "solution": "1, 2, and 3"},
    {"riddle": "I am a number. When you add me to myself, then divide by myself, you get 2. What am I?", "solution": "Any number except 0"},
    {"riddle": "I am a number. Multiply me by 3, subtract 4, then divide by 2, and you get 5. What am I?", "solution": "6"},
    {"riddle": "What is the smallest two-digit prime number?", "solution": "11"},
    {"riddle": "I am a number. If you subtract my ones digit from my tens digit, you get 3. If you multiply my digits together, you get 18. What am I?", "solution": "36"},
    {"riddle": "A rectangle has a length that is twice its width. If its area is 50 square units, what is its length?", "solution": "10"},
    {"riddle": "I am a number. If you divide me by 5 and add 7, you get 15. What am I?", "solution": "40"},
    {"riddle": "I am a number. If you add 7 to me and divide by 2, you get 9. What am I?", "solution": "11"}
]


with open("math_riddles.json", "w") as f:
    json.dump(data, f, indent=4)

!pip install -q unsloth
!pip install -q bitsandbytes transformers accelerate peft trl datasets huggingface_hub
import os
os.environ["HF_TOKEN"] =



from huggingface_hub import whoami
print(whoami())from unsloth import FastLanguageModel
import torch

model_id = "deepseek-ai/deepseek-math-7b-instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_id,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)

tokenizer.pad_token = tokenizer.eos_token
from transformers import AutoTokenizer


model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3977,
    use_rslora = False,
    loftq_config = None,
)from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deepseek-math-fine",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    num_train_epochs=6,
    push_to_hub=False,
    save_total_limit=2,
    report_to="none",
    run_name="deepseek-math-finetune"
)
from datasets import load_dataset

dataset = load_dataset("json", data_files="math_riddles.json")

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = [f"Riddle: {r} Answer: {a}{EOS_TOKEN}" for r, a in zip(examples["riddle"], examples["solution"])]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=2048
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

trainer.train()
import torch

def generate_riddle_and_answer():
    model.eval()

    prompt = "Riddle: "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in generated_text:
        parts = generated_text.split("Answer:", 1)
        riddle = parts[0].strip()
        answer = parts[1].strip()
    else:
        riddle = generated_text.strip()
        answer = "Could not detect an answer. Try adjusting parameters."

    return riddle, answer

riddle, answer = generate_riddle_and_answer()
print("Generated Riddle:", riddle)
print("Generated Answer:", answer)
