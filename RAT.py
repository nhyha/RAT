# Load the pre-trained model and tokenizer
model_name = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Define the ToT prompts and corresponding reasoning trees
tot_prompts = [
    {
        "prompt": "Problem: John has 5 apples. He gives 2 apples to Sarah. Later, John buys 3 more apples. How many apples does John have now?",
        "reasoning_tree": "Step 1: Initially, John has 5 apples.\nStep 2: John gives 2 apples to Sarah. So, John now has 5 - 2 = 3 apples.\nStep 3: John buys 3 more apples. So, John now has 3 + 3 = 6 apples.\nConclusion: John has 6 apples now.",
    },
    # Add more ToT prompts and reasoning trees
]

# Tokenize the prompts and reasoning trees
train_data = []
for item in tot_prompts:
    prompt_ids = tokenizer.encode(item["prompt"], return_tensors="pt")
    reasoning_tree_ids = tokenizer.encode(item["reasoning_tree"], return_tensors="pt")
    train_data.append((prompt_ids, reasoning_tree_ids))

# Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    save_steps=500,
    save_total_limit=2,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Fine-tune the model with LoRA
trainer.train()

# Save the fine-tuned model
model.save_pretrained("tot_lora_model")
