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
import os
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["gen", "denoise"], required=True)
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--valid_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--init_model", type=str, default="Salesforce/codet5-base")
parser.add_argument("--num_train_epochs", type=int, default=None)
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default=None)
args = parser.parse_args()

print(f"Loading model from: {args.init_model}")
tokenizer = AutoTokenizer.from_pretrained(args.init_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.init_model)

if args.task == "denoise":
    default_epochs = 3
    target_max_len = 64
elif args.task == "gen":
    default_epochs = 4
    target_max_len = 64

num_epochs = args.num_train_epochs if args.num_train_epochs else default_epochs



print(f"Loading data from: {args.train_file}")
df_train = pd.read_csv(args.train_file)
dataset_train = Dataset.from_pandas(df_train)

print(f"Loading data from: {args.valid_file}")
df_valid = pd.read_csv(args.valid_file)
dataset_valid = Dataset.from_pandas(df_valid)

def tokenize(example):
    model_input = tokenizer(
        example["content"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    labels = tokenizer(
        example["title"],
        truncation=True,
        padding="max_length",
        max_length=target_max_len
    )
    model_input["labels"] = labels["input_ids"]
    return model_input

print("Tokenizing...")
disable_progress_bar()
tokenized_dataset_train = dataset_train.map(tokenize)
tokenized_dataset_valid = dataset_valid.map(tokenize)
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

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=14,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=4,
    generation_max_length=64,
    save_strategy="epoch",
    eval_strategy="no",
    save_total_limit=2,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=100,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    report_to="none",
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
    hub_strategy="end"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(f"Starting training for task: {args.task} | Epochs: {num_epochs}")
import gc
gc.collect()
torch.cuda.empty_cache()
trainer.train()

if args.push_to_hub:
    
    login(token=secret_value)
    trainer.push_to_hub()
    print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")