import torch
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_from_disk

from utils import print_trainable_parameters, evaluate_model, prepare_prompt
import transformers

import json
import time

# Load dataset
dataset_split = load_from_disk('dataset_split')

slm_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_slm = AutoTokenizer.from_pretrained(slm_name, cache_dir="/Data/gabriel-mercier/slm_models", padding_side="left")
tokenizer_slm.pad_token = tokenizer_slm.eos_token

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'device : {device}')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type='nf4',
                            )
model_raw = AutoModelForCausalLM.from_pretrained(
    slm_name,
    cache_dir="/Data/gabriel-mercier/slm_models",
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"   
)

lora_config = LoraConfig(r=16, 
                        lora_alpha=32,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_dropout=0.05,
                        bias='none',
                        task_type="CAUSAL_LM")

model = get_peft_model(model_raw, lora_config)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

perc_param = print_trainable_parameters(model)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer_slm.eos_token_id
generation_config.eos_token_id = tokenizer_slm.eos_token_id
generation_config.do_sample = True

# Preprocess function
def generate_and_tokenize_prompt(data_point):
    full_prompt = prepare_prompt(data_point)+tokenizer_slm.eos_token 
    tokenized_full_prompt = tokenizer_slm(full_prompt, return_tensors='pt')
    labels = tokenized_full_prompt.input_ids.clone() 
    
    assistant_token = tokenizer_slm("Résumé concis et structuré", return_tensors='pt')['input_ids'][0]
   
    complement_token = tokenizer_slm("(100 mots maximum) :", return_tensors='pt')['input_ids'][0]
    
    T = tokenized_full_prompt['input_ids'].flatten()
    S = assistant_token.flatten()
    
    for i in range(len(T) - len(S) + 1):
        if torch.equal(T[i:i+len(S)], S):
            end_prompt_idx = i+len(S)   
    
    labels[:, :end_prompt_idx+len(complement_token)] = -100
    

    return {
        'input_ids': tokenized_full_prompt.input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': tokenized_full_prompt.attention_mask.flatten(),
    }

dataset_train = dataset_split["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
dataset_val = dataset_split["validation"].shuffle(seed=42).map(generate_and_tokenize_prompt)
dataset_test = dataset_split["test"]

dataset_train = dataset_train.remove_columns(["text", "summary"])
dataset_val = dataset_val.remove_columns(["text", "summary"])

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
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer_slm, model=model),
)

# Train model
start_time = time.time()
trainer.train()
end_training = time.time()

# Log training information
logs = trainer.state.log_history
train_losses = [log["loss"] for log in logs if "loss" in log]
eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

with open(f"./qwen_infos.json", "w") as f:
    json.dump({"train_losses": train_losses, "eval_losses": eval_losses, "perc_training":perc_param, "time_training":end_training-start_time}, f)


# Evaluate fine-tuned model
rouges_results, bert_results = evaluate_model(model, dataset_test, tokenizer_slm, device, generation_config)
results = {
    "rouge": rouges_results,
    "bert": bert_results
}
with open("evaluation_results_finetune.json", "w") as f:
    json.dump(results, f, indent=4)


# Evaluate raw model
model_raw = AutoModelForCausalLM.from_pretrained(
    slm_name,
    cache_dir="/Data/gabriel-mercier/slm_models",
)
model_raw.to(device)

rouges_results_raw, bert_results_raw = evaluate_model(model_raw, dataset_test)
results_raw = {
    "rouge": rouges_results_raw,
    "bert": bert_results_raw
}
with open("evaluation_results_raw.json", "w") as f:
    json.dump(results_raw, f, indent=4)
