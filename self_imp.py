import os
import argparse
import pandas as pd
from datasets import Dataset
import importlib
import transformers
importlib.reload(transformers)
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, Seq2SeqTrainingArguments
)
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from rouge_score import rouge_scorer
import torch
from rouge import Rouge 
rouge = Rouge()
parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--valid_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--init_model", type=str, default="Salesforce/codet5-base")

args = parser.parse_args()

print(f"Loading model from: {args.init_model}")
tokenizer = AutoTokenizer.from_pretrained(args.init_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.init_model)



print(f"Loading data from: {args.train_file}")
df_train = pd.read_csv(args.train_file)
dataset_train = Dataset.from_pandas(df_train)

print(f"Loading data from: {args.valid_file}")
df_valid = pd.read_csv(args.valid_file)
dataset_valid = Dataset.from_pandas(df_valid)



enable_progress_bar()

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(pred.strip().split(". ")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split(". ")) for label in decoded_labels]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(pred, ref) for pred, ref in zip(decoded_preds, decoded_labels)]

    rouge1_f1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores)
    rouge2_f1 = sum(s["rouge2"].fmeasure for s in scores) / len(scores)
    rougeL_f1 = sum(s["rougeL"].fmeasure for s in scores) / len(scores)

    return {
        "rouge1": rouge1_f1,
        "rouge2": rouge2_f1,
        "rougeL": rougeL_f1,
    }
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('multitask_seft.txt','w',encoding='utf-8') as f:
    for index, row in df_train.iterrows():
        #if index%1000 == 0: 
            #print(index)
        inputs = tokenizer(row['text'].strip(), return_tensors="pt", max_length=512, padding="max_length", truncation=True).to(DEVICE)
        summary_ids = model.generate(inputs["input_ids"], num_beams=20, min_length=2, max_length=48, num_return_sequences=20, top_p = 0.9)
        best_outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        scores0 = rouge.get_scores(best_outputs, row['title'].strip())
        best_score = scores0[0]['rouge-l']['f']
        for i in range(1,20):
            outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[i]
            scores = rouge.get_scores(outputs, row['title'].strip())
            if scores[0]['rouge-l']['f'] > best_score:
                best_outputs = outputs
                best_score = scores[0]['rouge-l']['f']   
        f.write(best_outputs+'\n')

print("Gen data for self-improve complete!")

import pandas as pd

# Specify the input CSV file path and input text file path
input_csv_path = args.train_file

input_txt_path = 'multitask_seft.txt'

df = pd.read_csv(input_csv_path)
a = []

# Read the text file and add its content as a new column
with open(input_txt_path, 'r', encoding = 'utf-8') as txt_file:
    # Read the lines from the text file
    lines = txt_file.readlines()
    for line in lines:
        a.append(line.strip())


df['title'] = [title
              for title in a]


# Specify the output CSV file path
output_csv_path = 'train_augment_all_final_f1.csv'

# Save the updated DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"New column 'text' added. Check the output file: {output_csv_path}")