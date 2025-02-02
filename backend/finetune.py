import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, pipeline
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

dataset = load_dataset("cognitivecomputations/samantha-data")
half_size = len(dataset['train']) // 2
dataset['train'] = dataset['train'].select(range(half_size))

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_prompt(example):
    user_turns = example["conversations"]["human"]
    ai_turns = example["conversations"]["gpt"]
    num_turns = min(len(user_turns), len(ai_turns))
    formatted_dialogue = "".join(
        f"<|user|> {user_turns[i]}\n<|assistant|> {ai_turns[i]}\n"
        for i in range(num_turns)
    )
    return {"text": f"<|system|> You are BUD, an AI designed for mental health support.\n{formatted_dialogue}"}

dataset = dataset.map(format_prompt, remove_columns=["id", "conversations"])

def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

model = prepare_model_for_kbit_training(model)

# This is setting up LoRA traingin , ye bhi dekh skti h
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked language modeling isse bhi dekhio
)

training_args = TrainingArguments(
    output_dir="./llama_samantha_bud",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    save_steps=1000,
    logging_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

# Saving 
model.save_pretrained("fine_tuned_llama_samantha_bud")
tokenizer.save_pretrained("fine_tuned_llama_samantha_bud")
model = model.merge_and_unload()
chatbot = pipeline(
    "text-generation",
    model="fine_tuned_llama_samantha_bud",
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Finallyyy inferencing
response = chatbot(
    "I'm feeling really stressed today. What should I do?",
    max_length=100,
    pad_token_id=tokenizer.eos_token_id
)
print(response[0]["generated_text"])