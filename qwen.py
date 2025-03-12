from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import evaluate
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import matplotlib.pyplot as plt
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import DatasetDict
import os
from utils import prepare_prompt
from transformers import DataCollatorForSeq2Seq
import transformers
from tqdm import tqdm
import numpy as np


dataset_raw = load_dataset('json', data_files='dataset_llm_generated.json')
dataset = dataset_raw.select_columns(["text", "summary"])

print(dataset)

split_train_temp = dataset["train"].train_test_split(test_size=0.4, seed=42)

split_valid_test = split_train_temp["test"].train_test_split(test_size=0.5, seed=42)

dataset_split = DatasetDict({
    "train": split_train_temp["train"],        
    "validation": split_valid_test["train"],      
    "test": split_valid_test["test"]              
})

print(dataset_split)

max_length = 2500

dataset_test = dataset_split['test'].filter(lambda x: len(x['text'].split()) <= max_length)

slm_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_slm = AutoTokenizer.from_pretrained(slm_name, cache_dir="/Data/gabriel-mercier/slm_models", padding_side="left")
tokenizer_slm.pad_token = tokenizer_slm.eos_token


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'device : {device}')
model = AutoModelForCausalLM.from_pretrained("./autoregressive_model")
model.to(device)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer_slm.eos_token_id
generation_config.eos_token_id = tokenizer_slm.eos_token_id
generation_config.do_sample = True



rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")

assistant_start = "Résumé concis et structuré (100 mots maximum) :"


def evaluate_model(model, dataset):
    summaries = [data_point['summary'] for data_point in dataset]
    predictions = []

    for data_point in tqdm(dataset):
        prompt = prepare_prompt(data_point, summary_included=False)
        encoding = tokenizer_slm(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config,
            )
            
        prediction = tokenizer_slm.decode(output[0], skip_special_tokens=True)
        response_start = prediction.find(assistant_start)
        # print(f"response start {response_start}")
        predictions.append(prediction[response_start:])
    # print(f"predictions {predictions}")
    rouge_results = rouge.compute(predictions=predictions, references=summaries)
    bert_results = bert_score.compute(predictions=predictions, references=summaries, lang="fr")

    # Moyenne des scores sur toutes les phrases
    bert_precision = np.mean(bert_results['precision'])
    bert_recall = np.mean(bert_results['recall'])
    bert_f1 = np.mean(bert_results['f1'])

    print(f"BERTScore - Precision: {bert_precision:.4f}, Recall: {bert_recall:.4f}, F1: {bert_f1:.4f}")
    print(f"ROUGEScores - {rouge_results}")
    print('\n')
    
    return rouge_results, {'Precision':bert_precision, 'Recall':bert_recall, 'F1':bert_f1}


rouges_results, bert_results = evaluate_model(model, dataset_test)


import json

results = {
    "rouge": rouges_results,
    "bert": bert_results
}

with open("evaluation_results_finetune.json", "w") as f:
    json.dump(results, f, indent=4)


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
