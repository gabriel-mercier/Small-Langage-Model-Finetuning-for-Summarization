# SMALL LANGUAGE MODEL FINETUNING FOR SUMMARIZATION
Gabriel Mercier and Ayoub Melliti

March 14th, 2025

# Code description
## `data_generation` folder : 

 

- notebook `generate_dataset.ipynb` used to generate the file `dataset_llm_generated.json` with Qwen2.5-32B-instruct. 

- `postprocess_dataset.ipynb` Post processes the dataset and splits int into train, validation and test dataset in `dataset_split`.

## Finetuning-Code :
- `t5.ipynb` Is a notebook which contains the code to finetune mT5 on the dataset, and evaluate it
- `t5.py` Contains the same code, and is used to run the finetuning and evaluation togerther with tmux.

- same purpose for`qwen.ipynb`, `qwen.py`.
- The code for finetuning mT5 is the same as T5. The only change needed is the variable `model_name`

## `result` folder :

### 