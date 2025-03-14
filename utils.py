import torch
import numpy as np
from tqdm import tqdm
import evaluate

# Mode teacher forcing
def prepare_prompt(data_point, summary_included=True):
    prompt = f"Résume précisément le texte suivant en français en 100 mots maximum. Concentre-toi sur les points essentiels sans ajouter d'opinions ni de commentaires. Évite les phrases inutiles et reformule les idées clairement.\n\nTexte :\n{data_point['text']}\n\nRésumé concis et structuré (100 mots maximum) :"
    if summary_included:
        prompt+=f"\n\n{data_point['summary']}"
    return prompt

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    text = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
   
    print(text)
    return text



def evaluate_model(model, dataset, tokenizer, device, generation_config):
    
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    
    assistant_start = "Résumé concis et structuré (100 mots maximum) :"
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