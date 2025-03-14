import torch
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from datasets import load_from_disk

from utils import print_trainable_parameters, evaluate_model
import transformers

import json
import time

r = 128
model_name = "t5-base" # "mt5-base"

# Load dataset
dataset_split = load_from_disk('dataset_split')

print(dataset_split)

cache_dir = "/Data/gabriel-mercier/slm_models"

# Load tokenizer
if model_name == "t5-base":
    final_name = "google-t5/t5-base"
elif model_name == "mt5-base":
    final_name = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(final_name, cache_dir=cache_dir)

# Configure BitsAndBytes
bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type='nf4',
                            )

# Load model
model_raw = AutoModelForSeq2SeqLM.from_pretrained(final_name, 
                                              cache_dir=cache_dir,
                                              trust_remote_code=True,
                                              quantization_config=bnb_config,
                                              device_map="auto")

# Configure LoRA
lora_config = LoraConfig(r=r, 
                            lora_alpha=2*r,
                            target_modules=["q", "k", "v", "o"],
                            lora_dropout=0.05,
                            bias='none',
                            task_type="SEQ_2_SEQ_LM")

# Apply LoRA to model
model = get_peft_model(model_raw, lora_config)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Configure generation settings
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.do_sample = True

# Print trainable parameters
perc_param = print_trainable_parameters(model)

# Preprocess function
def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], padding="max_length", max_length=2500, truncation=True)
    labels = tokenizer(examples["summary"], padding="max_length", max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess datasets
dataset_train = dataset_split["train"].map(preprocess_function)
dataset_val = dataset_split["validation"].map(preprocess_function)

# Remove unnecessary columns
dataset_train = dataset_train.remove_columns(["text", "summary"])
dataset_val = dataset_val.remove_columns(["text", "summary"])

print(dataset_train)
print(dataset_val)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=1,
    evaluation_strategy="epoch",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

# Trainer setup
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)

# Train model
start_time = time.time()
trainer.train()
end_training = time.time()

# Save model
trainer.save_model(f"./t5_r{r}")

# Log training information
logs = trainer.state.log_history
train_losses = [log["loss"] for log in logs if "loss" in log]
eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

with open(f"./t5_r{r}_infos.json", "w") as f:
    json.dump({"train_losses": train_losses, "eval_losses": eval_losses, "perc_training":perc_param, "time_training":end_training-start_time}, f)

# Load fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained(f"./t5_r{r}")
model.to(device)

# Load test dataset
dataset_test = dataset_split['test']

# Evaluate fine-tuned model
rouges_results_finetune, bert_results_finetune = evaluate_model(model, dataset_test, tokenizer, device, generation_config)

results_finetune = {
    "rouge": rouges_results_finetune,
    "bert": bert_results_finetune
}

with open(f"t5_evaluation_results_finetune_r{r}.json", "w") as f:
    json.dump(results_finetune, f, indent=4)

# Evaluate raw model
rouges_results_raw, bert_results_raw = evaluate_model(model_raw, dataset_test, tokenizer, device, generation_config)

results_raw = {
    "rouge": rouges_results_raw,
    "bert": bert_results_raw
}

with open("t5_evaluation_results_raw.json", "w") as f:
    json.dump(results_raw, f, indent=4)
