import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import evaluate
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from datasets import load_from_disk

import matplotlib.pyplot as plt
from utils import prepare_prompt, print_trainable_parameters
import transformers
from tqdm import tqdm

import numpy as np
import json
import time


lora_finetune = True
r = 8

dataset_split = load_from_disk('dataset_split')

print(dataset_split)

cache_dir = "/Data/gabriel-mercier/slm_models"

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", cache_dir=cache_dir)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type='nf4',
                            )
if lora_finetune:
    model_raw = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", 
                                              cache_dir=cache_dir,
                                              trust_remote_code=True,
                                              quantization_config=bnb_config,
                                              device_map="auto")
else:
    model_raw = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base",
                                          cache_dir=cache_dir,
                                          trust_remote_code=True,
                                          device_map="auto")
    


if lora_finetune:
    lora_config = LoraConfig(r=r, 
                            lora_alpha=2*r,
                            target_modules=["q", "k", "v", "o"],
                            lora_dropout=0.05,
                            bias='none',
                            task_type="SEQ_2_SEQ_LM")

    model = get_peft_model(model_raw, lora_config)

else:
    model = model_raw
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.do_sample = True

perc_param = print_trainable_parameters(model)

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], padding="max_length", max_length=2500, truncation=True)
    labels = tokenizer(examples["summary"], padding="max_length", max_length=150, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset_train = dataset_split["train"].map(preprocess_function)
dataset_val = dataset_split["validation"].map(preprocess_function)

dataset_train = dataset_train.remove_columns(["text", "summary"])
dataset_val = dataset_val.remove_columns(["text", "summary"])

print(dataset_train)
print(dataset_val)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


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

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
)


start_time = time.time()
trainer.train()
end_training = time.time()

trainer.save_model(f"./t5_r{r}")

logs = trainer.state.log_history
train_losses = [log["loss"] for log in logs if "loss" in log]
eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

with open(f"./t5_r{r}_infos.json", "w") as f:
    json.dump({"train_losses": train_losses, "eval_losses": eval_losses, "perc_training":perc_param, "time_training":end_training-start_time}, f)


model = AutoModelForSeq2SeqLM.from_pretrained(f"./t5_r{r}")
model.to(device)

dataset_test = dataset_split['test']

rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")
assistant_start = "Résumé concis et structuré (100 mots maximum) :"

def evaluate_model(model, dataset):
    summaries = [data_point['summary'] for data_point in dataset]
    predictions = []

    for data_point in tqdm(dataset):
        prompt = prepare_prompt(data_point, summary_included=False)
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config,
            )
            
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        
        start_index = prediction.find(assistant_start)
        if start_index != -1:
            response_start = start_index + len(assistant_start)
        else:
            response_start = -1 
        predictions.append(prediction[response_start+1:])

    rouge_results = rouge.compute(predictions=predictions, references=summaries)
    bert_results = bert_score.compute(predictions=predictions, references=summaries, lang="fr")

    bert_precision = np.mean(bert_results['precision'])
    bert_recall = np.mean(bert_results['recall'])
    bert_f1 = np.mean(bert_results['f1'])

    print(f"BERTScore - Precision: {bert_precision:.4f}, Recall: {bert_recall:.4f}, F1: {bert_f1:.4f}")
    print(f"ROUGEScores - {rouge_results}")
    print('\n')
    
    return rouge_results, {'Precision':bert_precision, 'Recall':bert_recall, 'F1':bert_f1}

rouges_results_finetune, bert_results_finetune = evaluate_model(model, dataset_test)

results_finetune = {
    "rouge": rouges_results_finetune,
    "bert": bert_results_finetune
}

with open(f"t5_evaluation_results_finetune_r{r}.json", "w") as f:
    json.dump(results_finetune, f, indent=4)


'''rouges_results_raw, bert_results_raw = evaluate_model(model_raw, dataset_test)

results_raw = {
    "rouge": rouges_results_raw,
    "bert": bert_results_raw
}

with open("t5_evaluation_results_raw.json", "w") as f:
    json.dump(results_raw, f, indent=4)'''
