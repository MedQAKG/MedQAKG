import json
import logging
import os

import argparse
import collections
import math
import random
import time
import re
import string
import sys
from io import open

from question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def normalize_answer(s):

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


with open('../../../simple_transformer/train_data.json', 'r') as f:
    train_data = json.load(f)

# train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]


with open('../../../simple_transformer/val_data.json', 'r') as f:
    dev_data = json.load(f)


train_args = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'max_seq_length': 512,
    'doc_stride': 384,
    'output_dir': "roberta_att_2_filter_mashqa/",
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 8,
}

# model = QuestionAnsweringModel("bert", "bert-base-uncased", args=train_args)
model = QuestionAnsweringModel("roberta", "roberta-large", args=train_args)



# Train the model
model.train_model(train_data, eval_data=dev_data)

# Evaluate the model
result, texts = model.eval_model(dev_data)

# print("1111111 TEXTS !!!!!!!!!!!!!!")
# print(texts)

print("11111111111!!!!!!!!!!@@@@@@@  roberta with TransE @@@@@@@##########")
print(result)



exact_scores = {}
f1_scores = {}
for k1,v1 in texts.items():
    if k1 =='correct_text':
        pass
    else:
        for k2,v2 in v1.items():
            exact_scores[k2] =compute_exact(normalize_answer(v2['truth']),normalize_answer(v2['predicted']))
            f1_scores[k2] =compute_f1(v2['truth'],v2['predicted'])        


em=result['correct']
f1=result['correct']

for k,v in exact_scores.items():
    em= em + v


for k,v in f1_scores.items():
    f1= f1 + v


final_em= em/(result['correct']+result['incorrect']+result['similar'])
final_f1= f1/(result['correct']+result['incorrect']+result['similar'])

print("!!!!!!! Exact Match !!!!!!!!!")
print(final_em)

print("!!!!!!! F1 Score !!!!!!!!!")
print(final_f1)

