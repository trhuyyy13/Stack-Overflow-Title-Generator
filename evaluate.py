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
import nltk.translate.gleu_score as gleu
import string
from tqdm import tqdm
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument("--ranking", type=bool, required=True)
parser.add_argument("--test_folder", type=str, required=True)
parser.add_argument("--init_model", type=str, default="Salesforce/codet5-base")
args = parser.parse_args()

print(f"Loading model from: {args.init_model}")
tokenizer = AutoTokenizer.from_pretrained(args.init_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.init_model)

from rouge import FilesRouge

def get_rouge(hyp_path, ref_path):
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
    return scores

lang = ['c_sharp', 'python','java','js']

import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter


def build_bi_gram_vector(predictions):
    bi_gram_predictions = [] 
    bi_grams = []
    for predict in predictions:
        bi_gram_result = list(zip(*[predict[i:] for i in range(2)]))
        bi_grams += bi_gram_result
        bi_gram_predictions.append(bi_gram_result)
    bi_gram_count = Counter(bi_grams) 
    bi_gram_ids = {} 
    for i, bi_gram_key in enumerate(bi_gram_count.keys()):
        bi_gram_ids[bi_gram_key] = i
    prediction_vectors = []
    for p in bi_gram_predictions:
        initial = np.zeros(len(bi_gram_count.keys()))
        for b in p:
            initial[bi_gram_ids[b]] = bi_gram_count[b]
        prediction_vectors.append(initial)
    return np.array(prediction_vectors)


def scoring_prediction_bi_gram(predictions):
    with_weight = []
    bi_gram_predictions = []
    bi_grams = []
    for predict in predictions:
        # print(predict)
        bi_gram_result = list(zip(*[predict[i:] for i in range(2)]))
        # print(bi_gram_result)
        bi_grams += bi_gram_result
        bi_gram_predictions.append(bi_gram_result)
    bi_gram_count = Counter(bi_grams)
    for i in range(len(predictions)):
        weight = sum([bi_gram_count[bg] for bg in bi_gram_predictions[i]])
        with_weight.append((predictions[i], i, weight))
    with_weight_sorted = sorted(with_weight, key=itemgetter(2), reverse=True)
    # print(with_weight_sorted)
    return with_weight_sorted

def scoring_prediction_bi_gram_average(predictions):
    with_weight = []
    bi_gram_predictions = []
    bi_grams = []
    for predict in predictions:
        # print(predict)
        bi_gram_result = list(zip(*[predict[i:] for i in range(2)]))
        # print(bi_gram_result)
        bi_grams += bi_gram_result
        bi_gram_predictions.append(bi_gram_result)
    bi_gram_count = Counter(bi_grams)
    for i in range(len(predictions)):
        weight = sum([bi_gram_count[bg] for bg in bi_gram_predictions[i]])
        weight = weight/len(bi_gram_predictions[i])
        with_weight.append((predictions[i], i, weight))
    with_weight_sorted = sorted(with_weight, key=itemgetter(2), reverse=True)
    # print(with_weight_sorted)
    return with_weight_sorted
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
enable_progress_bar()
if args.ranking:
    num_gen=30 
else:
    num_gen=1
for i in range (len(lang)):
    df = pd.read_csv(args.test_folder + '/multitask-dataset/test_multitask_'+lang[i]+'.csv')
    print(lang[i])
    with open(f'{lang[i]}_{num_gen}_candidate.txt', 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            inputs = tokenizer(row['content'].strip(), return_tensors="pt", max_length=512, padding="max_length", truncation=True).to(DEVICE)
            # summary_ids = model.generate(inputs["input_ids"], num_beams=j, num_return_sequences = j,  min_length=2, max_length=48, top_p = 0.9, top_k = 5)
            summary_ids = model.generate(inputs["input_ids"], num_beams=num_gen, num_return_sequences = num_gen,  min_length=2, max_length=48, top_p = 0.9, top_k = 5, length_penalty = 0.0)

             
            for k in range(num_gen):
                # summary_ids = model.generate(inputs, num_beams=4, min_length=2, max_length=48, top_p = 0.1)

                best_outputs = tokenizer.decode(summary_ids[k], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print(best_outputs)

                f.write(best_outputs+'\n')
                
for type_lang in lang:
    with open(type_lang+'_'+str(num_gen)+'_candidate.txt','r',encoding='utf-8') as f, open(type_lang+'_'+str(num_gen)+'_ranking.txt','w',encoding='utf-8') as f1:
            i = 0
            ans = []
            ans1 = []
            for line in f:
                i += 1
                x = line.strip().split()
                ans.append(x)
                if i<num_gen: 
                    continue
                else:
                    ans1.append(ans)
                    i = 0
                    ans = []
            for predict in ans1:
                # print(predict)
                scoring_prediction_bi_gram(predict)
        
                # break
                f1.write(' '.join(scoring_prediction_bi_gram(predict)[0][0])+'\n')


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)


def computeBleu1_to_4(reference_list, candidate_list):
    smooth = SmoothingFunction().method1
    bleu1_sum = bleu2_sum = bleu3_sum = bleu4_sum = bleuA_sum = 0

    for (ref, cand) in zip(reference_list, candidate_list):

        tokens_real = ref.split(' ')
        tokens_pred = cand.split(' ')

        if cand == '':
            bleu1_score = bleu2_score = bleu3_score = bleu4_score = bleuA_score = 0

        else:
            bleu1_score = sentence_bleu([tokens_real], tokens_pred, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
            bleu2_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.0, 1.0, 0.0, 0.0), smoothing_function=smooth)
            bleu3_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.0, 0.0, 1.0, 0.0), smoothing_function=smooth)
            bleu4_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.0, 0.0, 0.0, 1.0), smoothing_function=smooth)
            bleuA_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        bleu1_sum += bleu1_score
        bleu2_sum += bleu2_score
        bleu3_sum += bleu3_score
        bleu4_sum += bleu4_score
        bleuA_sum += bleuA_score

    return {
    "BLEU_A": round(bleuA_sum / len(reference_list), 3) * 100,
    "BLEU_1": round(bleu1_sum / len(reference_list), 3) * 100,
    "BLEU_2": round(bleu2_sum / len(reference_list), 3) * 100,
    "BLEU_3": round(bleu3_sum / len(reference_list), 3) * 100,
    "BLEU_4": round(bleu4_sum / len(reference_list), 3) * 100,
}

for type_lang in lang:
    df = pd.read_csv(args.test_folder+"/test_multitask_"+type_lang+".csv")
    df['title'] = df['title'].fillna("").astype(str)
    df['title'].astype(str).to_csv("target_"+type_lang+".txt", index=False, header=False)
        
from rouge import FilesRouge

def get_rouge(hyp_path, ref_path):
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
    return scores

results = []
for type_lang in lang:
    scores_rouge = get_rouge(type_lang+"_"+str(num_gen)+"_ranking.txt", "target_"+type_lang+".txt")
    with open('target_'+type_lang+'.txt','r',encoding='utf-8') as f, open(type_lang+'_'+str(num_gen)+'_ranking.txt','r',encoding='utf-8') as f1:
        reference = [line.strip() for line in f.readlines()]
        hypothesis = [line.strip() for line in f1.readlines()]
    
    gleu_score = score_gleu(reference, hypothesis)
    bleu = computeBleu1_to_4(reference,hypothesis)
    result = {
    "Language": type_lang,
    "ROUGE-1": round(scores_rouge["rouge-1"]["f"], 3),
    "ROUGE-2": round(scores_rouge["rouge-2"]["f"], 3),
    "ROUGE-L": round(scores_rouge["rouge-l"]["f"], 3),
    "GLEU": round(gleu_score, 3),
    "BLEU_1": round(bleu["BLEU_1"], 3),
    "BLEU_2": round(bleu["BLEU_2"], 3),
    "BLEU_3": round(bleu["BLEU_3"], 3),
    "BLEU_4": round(bleu["BLEU_4"], 3),
    "BLEU_A": round(bleu["BLEU_A"], 3),
    }
    results.append(result)

df_result = pd.DataFrame(results)
df_result.to_csv('evaluation_results.csv', index=False)