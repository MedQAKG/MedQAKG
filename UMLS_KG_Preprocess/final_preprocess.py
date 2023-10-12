import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english')

from load_covidqa_dic import*
from create_kg import*
from biobert_embedding_cosine_sim import*
from preprocess import*


pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')


## read covidqa-data
with open('../../Data/train_data.json','r') as f:
	covid_train_data=json.load(f)

train_data=[]

for item,val in enumerate(covid_train_data):
	print("context no: ",item)
	context_kg = context_kg_triple(covid_train_data[item]["context"])
	qq=[]
	for qa in covid_train_data[item]["qas"]:
		answers=[]
		answers.append({"text":qa["answers"][0]["text"], "answer_start":qa["answers"][0]["answer_start"]})
		question_text =qa["question"]
		triple=sorted_triple(question_text,context_kg,pubmed_bert_model,pubmed_bert_tokenizer)
		q = {
			"id":qa["id"],
			"is_impossible": qa["is_impossible"],
			"question": qa["question"],
			"kg_triplets": triple,
			"answers": answers
		}
		qq.append(q)
	train ={
			"context":covid_train_data[item]["context"],
			"qas":qq
	}
	train_data.append(train)

	if len(train_data)%500==0:
		with open('MashQA_kg_train_data_first_500.json', 'w') as fp:
		    json.dump(train_data, fp, indent=4)
		break

